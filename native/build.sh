#!/bin/bash
# Build the system-audio helper (macOS 14.4+, needs Xcode command line tools).
cd "$(dirname "$0")"
mkdir -p bin
swiftc -swift-version 5 -O systap.swift -o bin/systap && echo "built native/bin/systap"
