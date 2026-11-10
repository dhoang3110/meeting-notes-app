# Meeting Notes — ghi âm & tóm tắt cuộc họp chạy local

Tool chạy trên Mac (Apple Silicon): ghi âm cuộc họp, chuyển giọng nói thành text bằng Parakeet ngay trên máy (chỉ tiếng Anh), rồi tóm tắt thành **Summary, Action Items, Decisions, Open Questions** bằng Ollama (local, mặc định) hoặc Claude API. Kết quả lưu thành file Markdown trong `~/MeetingNotes`.

```
Mic + System audio ──► Parakeet (local, mlx) ──► Ollama / Claude ──► notes.md
```

Mỗi cuộc họp là một thư mục:

```
~/MeetingNotes/2026-11-03_103100_QA-strategy-sync/
  audio.flac        # bản ghi âm (tự xoá sau khi tóm tắt xong; giữ lại bằng KEEP_AUDIO=1 hoặc "keep_audio": true trong meta.json)
  transcript.json   # transcript có timestamp
  notes.json        # dữ liệu tóm tắt dạng JSON
  notes.md          # ghi chú cuối cùng
  meta.json
```

---

## 1. Cài đặt (làm một lần)

### 1.1 Công cụ cơ bản

```bash
# Homebrew nếu chưa có: https://brew.sh
brew install python@3.12 ffmpeg
```

`ffmpeg` cần cho tính năng upload file ghi âm có sẵn (m4a, mp4, mp3…).

### 1.2 Thu âm thanh của người khác trong cuộc họp (BlackHole)

macOS không cho app ghi trực tiếp âm thanh phát ra loa. Cần một thiết bị âm thanh ảo:

```bash
brew install blackhole-2ch
```

Sau đó khởi động lại Mac (hoặc logout/login), rồi:

1. Mở **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup").
2. Bấm **+** ở góc dưới trái → **Create Multi-Output Device**.
3. Tick **BlackHole 2ch** và **loa/tai nghe bạn hay dùng** (ví dụ "MacBook Pro Speakers" hoặc AirPods). Đặt tai nghe làm *Primary Device*, bật *Drift Correction* cho BlackHole.
4. Đổi tên thành `Meeting Output`.

Cách dùng khi họp, chọn **một** trong hai:

- **Cách khuyên dùng:** Trong Zoom/Teams → Settings → Audio → *Speaker* chọn `Meeting Output`. Chỉ âm thanh cuộc họp đi vào BlackHole, phần còn lại của Mac không bị ảnh hưởng.
- Hoặc đổi output của cả Mac sang `Meeting Output` (System Settings → Sound). Lưu ý: khi dùng Multi-Output Device, phím tăng/giảm âm lượng của Mac không hoạt động; chỉnh âm lượng trong app họp.

Với Google Meet trên trình duyệt: đổi output của cả Mac, hoặc trong Meet → Settings → Audio → Speakers chọn `Meeting Output` (Chrome hỗ trợ).

> Không cài BlackHole vẫn dùng được, nhưng tool chỉ ghi được mic. Nếu bạn đeo tai nghe thì sẽ không nghe được người khác trong bản ghi.

### 1.3 Ollama (nếu muốn tóm tắt hoàn toàn local)

```bash
brew install ollama
ollama serve &          # hoặc mở app Ollama
ollama pull qwen2.5:14b # ~9 GB, cần Mac 16 GB RAM trở lên
```

Máy 8 GB RAM: dùng `qwen2.5:7b` và sửa `OLLAMA_MODEL` trong `.env`. Máy 32 GB+: có thể thử `qwen2.5:32b` cho chất lượng tốt hơn.

### 1.4 Claude API (tuỳ chọn, thay cho Ollama)

Tạo API key tại https://console.anthropic.com, rồi điền `ANTHROPIC_API_KEY` vào `.env`. Mặc định app vẫn dùng Ollama; chọn "Claude API" trong dropdown Summary khi cần.

> Trước khi gửi nội dung họp của công ty lên Claude API, hãy kiểm tra chính sách bảo mật dữ liệu của công ty.

---

## 2. Chạy

```bash
cd meeting-notes
cp .env.example .env     # sửa OLLAMA_MODEL, ANTHROPIC_API_KEY… nếu cần
./run.sh
```

Lần đầu `run.sh` sẽ tạo virtualenv và cài thư viện (vài phút). Trình duyệt sẽ tự mở `http://127.0.0.1:8765`.

Lần đầu bấm **Start recording**, macOS sẽ hỏi quyền **Microphone** cho Terminal (hoặc iTerm). Chọn *Allow*. Nếu lỡ từ chối: System Settings → Privacy & Security → Microphone → bật cho Terminal, rồi chạy lại.

Lần đầu transcribe, Parakeet sẽ tải model (~2.5 GB) từ Hugging Face. Các lần sau dùng bản đã tải.

---

## 3. Sử dụng

1. **Microphone:** chọn mic của bạn (MacBook mic, AirPods…).
2. **System audio:** chọn `BlackHole 2ch` (tool tự chọn nếu tìm thấy).
3. Đặt tên cuộc họp (không bắt buộc), chọn Ollama (mặc định) hoặc Claude.
4. Bấm **● Start recording**. Hai thanh level sẽ nhảy khi có âm thanh; nếu thanh BlackHole không nhảy khi người khác nói, xem lại bước 1.2.
5. Hết họp bấm **■ Stop & summarise**. Tool sẽ transcribe rồi tóm tắt; cuộc họp 1 tiếng mất khoảng 2–5 phút trên M1/M2/M3.

Các nút khác:

- **Upload a recording…**: tóm tắt một file ghi âm/ghi hình có sẵn (m4a, mp3, mp4, wav…).
- **Re-summarise**: tóm tắt lại từ transcript có sẵn, ví dụ để thử model Ollama khác. Nếu lần trước bị lỗi ở bước transcribe, nút này sẽ chạy lại từ file audio.
- **Copy Markdown**: copy ghi chú để dán vào Notion, Confluence, Slack…
- **Show in Finder**: mở thư mục chứa ghi chú.

> Hãy thông báo cho mọi người trước khi ghi âm. Ở NSW, luật thường yêu cầu sự đồng ý của các bên khi ghi âm cuộc trò chuyện riêng.

---

## 4. Tuỳ chỉnh (`.env`)

| Biến | Ý nghĩa |
|---|---|
| `NOTES_LANGUAGE` | Ngôn ngữ của ghi chú, ví dụ `English` hoặc `Vietnamese` |
| `TRANSCRIBE_MODEL` | Speech-to-text engine, mặc định `parakeet:mlx-community/parakeet-tdt-0.6b-v2` (chỉ tiếng Anh) |
| `OLLAMA_MODEL` | Model local, ví dụ `qwen2.5:14b`, `llama3.1:8b` |
| `LIVE_TRANSCRIBE` | `1` (default): transcribe while recording, so the transcript is ready right after Stop. `0`: transcribe only after the recording ends |
| `LIVE_CHUNK_SEC` | Seconds of audio per live chunk (default `10`) |
| `LIVE_PARTIAL_SEC` | Show grey provisional captions every N seconds (default `2`, `0` = off) |
| `MEETING_DETECT` / `MEETING_DETECT_APPS` | `1` (default): hiện hộp thoại "Record?" khi Teams bắt đầu dùng mic (cần macOS 14.4+). Thêm app khác bằng bundle id, ví dụ `com.microsoft.teams,us.zoom.xos` |
| `MEETING_AUTO_STOP` / `MEETING_END_GRACE_SEC` | `1` / `40` (mặc định): tự dừng và tạo summary khi Teams nhả mic quá 40 giây, đoạn im lặng sau lúc nhả mic được cắt bỏ (áp dụng cho bản ghi bắt đầu từ hộp thoại "Record?" hoặc bấm Start trong lúc Teams đang dùng mic; mute không làm dừng). Nếu Teams dùng mic lại sớm hơn, app hỏi đây là cuộc họp mới hay cùng cuộc họp |
| `NOTES_DIR` | Thư mục lưu ghi chú (có thể trỏ vào Obsidian vault hoặc iCloud Drive) |

---

## 5. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| Không thấy BlackHole trong danh sách | Khởi động lại Mac sau khi cài, rồi reload trang |
| Bản ghi chỉ có giọng mình | Zoom/Teams chưa xuất âm thanh ra `Meeting Output` |
| Bản ghi im lặng hoàn toàn | Terminal chưa có quyền Microphone |
| Lỗi `ConnectionError` với Ollama | Chưa chạy `ollama serve`, hoặc chưa `ollama pull` model |
| Ollama tóm tắt kém hoặc thiếu ý | Dùng model Ollama lớn hơn, hoặc dùng Claude |

---

## 6. Giới hạn hiện tại

- Chưa phân biệt người nói (speaker diarization). Transcript không ghi "ai nói gì", nên owner của action item chỉ được điền khi nội dung cuộc họp nói rõ.
- Ghi âm hai thiết bị riêng rồi ghép lại: với cuộc họp rất dài (2 giờ+) có thể lệch vài trăm ms giữa hai nguồn, không ảnh hưởng đến việc tóm tắt.
- Đừng để Mac sleep trong lúc ghi âm (`caffeinate -d` trong một tab Terminal khác nếu cần).
