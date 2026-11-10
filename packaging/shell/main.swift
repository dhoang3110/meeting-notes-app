// Native shell for Meeting Notes: starts the bundled Python server on a free port and shows its
// UI in a WKWebView. The shell is the "responsible process" for macOS privacy prompts, so the
// microphone / system-audio permission dialogs name Meeting Notes, not Terminal or Python.
// Closing the window keeps the server running (meeting detection); Quit stops it.
import Cocoa
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, WKUIDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var server: Process?
    var port = 0

    func applicationDidFinishLaunching(_ n: Notification) {
        buildMenu()
        web = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        web.uiDelegate = self
        web.navigationDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1180, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Meeting Notes"
        window.contentView = web
        window.setFrameAutosaveName("main")
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        web.loadHTMLString("<body style='font:15px -apple-system;color:#888;display:flex;align-items:center;justify-content:center;height:90vh'>Starting Meeting Notes…</body>", baseURL: nil)
        startServer()
    }

    // MARK: server
    func freePort() -> Int {
        let s = socket(AF_INET, SOCK_STREAM, 0)
        var a = sockaddr_in(); a.sin_family = sa_family_t(AF_INET); a.sin_addr.s_addr = inet_addr("127.0.0.1"); a.sin_port = 0
        a.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafePointer(to: &a) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(s, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &a) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(s, $0, &len) } }
        close(s)
        return Int(UInt16(bigEndian: a.sin_port))
    }

    func startServer() {
        let res = Bundle.main.resourcePath!
        port = freePort()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: res + "/server/server")
        var env = ProcessInfo.processInfo.environment
        env["PORT"] = String(port)
        env["SYSTAP_BIN"] = res + "/bin/systap"
        p.environment = env
        let logDir = NSHomeDirectory() + "/Library/Logs/MeetingNotes"
        try? FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: logDir + "/server.log", contents: nil)
        if let h = FileHandle(forWritingAtPath: logDir + "/server.log") { p.standardOutput = h; p.standardError = h }
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                if self?.server === proc && proc.terminationStatus != 0 {
                    self?.web.loadHTMLString("<body style='font:15px -apple-system;padding:40px'>The Meeting Notes server stopped (exit \(proc.terminationStatus)).<br>See ~/Library/Logs/MeetingNotes/server.log</body>", baseURL: nil)
                }
            }
        }
        do { try p.run() } catch {
            web.loadHTMLString("<body style='font:15px -apple-system;padding:40px'>Could not start the server: \(error)</body>", baseURL: nil)
            return
        }
        server = p
        waitUntilReady(deadline: Date().addingTimeInterval(90))
    }

    func waitUntilReady(deadline: Date) {
        let url = URL(string: "http://127.0.0.1:\(port)/api/status")!
        URLSession.shared.dataTask(with: url) { [weak self] _, resp, _ in
            guard let self = self else { return }
            if (resp as? HTTPURLResponse)?.statusCode == 200 {
                DispatchQueue.main.async { self.web.load(URLRequest(url: URL(string: "http://127.0.0.1:\(self.port)/")!)) }
            } else if Date() < deadline, self.server?.isRunning == true {
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.4) { self.waitUntilReady(deadline: deadline) }
            }
        }.resume()
    }

    func stopServer() {
        guard let p = server, p.isRunning else { return }
        server = nil
        p.terminate()
        let t = Date().addingTimeInterval(5)
        while p.isRunning && Date() < t { Thread.sleep(forTimeInterval: 0.1) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
    }

    func applicationWillTerminate(_ n: Notification) { stopServer() }
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { false }
    func applicationShouldHandleReopen(_ s: NSApplication, hasVisibleWindows f: Bool) -> Bool {
        window.makeKeyAndOrderFront(nil); return true
    }

    // MARK: JavaScript dialogs (the UI uses alert/confirm)
    func webView(_ w: WKWebView, runJavaScriptAlertPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo, completionHandler c: @escaping () -> Void) {
        let a = NSAlert(); a.messageText = m; a.addButton(withTitle: "OK"); a.runModal(); c()
    }
    func webView(_ w: WKWebView, runJavaScriptConfirmPanelWithMessage m: String, initiatedByFrame f: WKFrameInfo, completionHandler c: @escaping (Bool) -> Void) {
        let a = NSAlert(); a.messageText = m; a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
        c(a.runModal() == .alertFirstButtonReturn)
    }
    func webView(_ w: WKWebView, runJavaScriptTextInputPanelWithPrompt p: String, defaultText d: String?, initiatedByFrame f: WKFrameInfo, completionHandler c: @escaping (String?) -> Void) {
        let a = NSAlert(); a.messageText = p; a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
        let f = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24)); f.stringValue = d ?? ""; a.accessoryView = f
        c(a.runModal() == .alertFirstButtonReturn ? f.stringValue : nil)
    }
    // External links open in the default browser
    func webView(_ w: WKWebView, decidePolicyFor nav: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let u = nav.request.url, nav.navigationType == .linkActivated, u.host != "127.0.0.1" {
            NSWorkspace.shared.open(u); decisionHandler(.cancel)
        } else { decisionHandler(.allow) }
    }
    func webView(_ w: WKWebView, createWebViewWith c: WKWebViewConfiguration, for a: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let u = a.request.url { NSWorkspace.shared.open(u) }
        return nil
    }

    // MARK: menu (Edit menu is required for copy/paste in text fields)
    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu(); appItem.submenu = appMenu
        appMenu.addItem(withTitle: "Quit Meeting Notes", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edit"); editItem.submenu = edit
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let winItem = NSMenuItem(); main.addItem(winItem)
        let win = NSMenu(title: "Window"); winItem.submenu = win
        win.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        NSApp.mainMenu = main
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
signal(SIGTERM, SIG_IGN)
let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termSource.setEventHandler { NSApp.terminate(nil) }   // `kill` / logout stop the server too
termSource.resume()
app.run()
