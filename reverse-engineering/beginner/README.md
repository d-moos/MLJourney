# Beginner Course: Foundations & Practical Patching

Welcome to the **Beginner** course on Windows binary reverse engineering. This course takes you from zero reversing experience to confidently analyzing and patching simple binaries.

---

## Who This Course Is For

You are an experienced software engineer who:

- Is comfortable programming in Rust, C, or C++
- Has a basic idea of assembly (registers, stack, instructions) but hasn't done real reversing
- Understands general Windows usage but not deep internals
- Wants to learn reverse engineering for security research, game modding, or personal curiosity

---

## What You'll Learn

By the end of this course, you will be able to:

1. **Set up a safe, reproducible reversing environment** with all necessary tools
2. **Understand the PE format** (EXE/DLL structure, sections, imports, exports)
3. **Read x86-64 assembly** comfortably (conditionals, loops, function calls, calling conventions)
4. **Perform static analysis** with Binary Ninja (navigate functions, find strings, trace control flow)
5. **Perform dynamic analysis** with x64dbg (breakpoints, stepping, memory inspection)
6. **Patch binaries** to change behavior (bypass checks, modify logic)
7. **Hook functions** in DLLs (understand and apply inline hooks)
8. **Reverse and patch a complete application** you built yourself

---

## Prerequisites

Before starting, ensure you have:

- Access to a **Windows 10/11 VM** with admin rights
- Ability to install software (Binary Ninja, x64dbg, etc.)
- **Rust** or a **C/C++ compiler** (MSVC, clang, or MinGW) installed
- Basic familiarity with command-line tools

---

## Quick Installation

We provide an automated PowerShell script to install all required tools:

```powershell
# Run PowerShell as Administrator, then:
powershell -ExecutionPolicy Bypass -File install_tools.ps1
```

The script will attempt to install all tools and report any failures at the end with manual installation links.

---

## Tools You'll Install

| Tool | Purpose |
|------|---------|
| **Binary Ninja** | Disassembler and decompiler for static analysis |
| **x64dbg** | Debugger for dynamic analysis |
| **PE-bear** or **CFF Explorer** | PE header inspection |
| **Process Explorer** | View running processes and loaded modules |
| **ProcMon** | Monitor filesystem and registry activity |
| **Python 3** | Scripting and automation (used more in later courses) |

---

## Course Duration

**Estimated time:** 10–15 hours over 3–4 weeks

Each lesson includes reading, hands-on exercises, and solutions. Spend time on exercises—they're where the real learning happens.

---

## Table of Contents

1. [Lesson 1: Setting up a Safe Reversing Lab](lesson_01_lab_setup.md)
   - Create an isolated VM environment
   - Install and configure all tools
   - Build your first target binary

2. [Lesson 2: x86-64 Assembly Refresher for Reversers](lesson_02_x86_64_refresher.md)
   - Registers, flags, and common instructions
   - Windows x64 calling convention
   - Recognizing compiler patterns

3. [Lesson 3: PE Format Basics](lesson_03_pe_format_basics.md)
   - DOS header, PE header, sections
   - Imports, exports, and the IAT
   - RVA vs file offset conversions

4. [Lesson 4: Static Analysis with Binary Ninja](lesson_04_binary_ninja_static_analysis.md)
   - Loading and navigating binaries
   - Finding functions, strings, and cross-references
   - Reading decompiled output

5. [Lesson 5: Dynamic Analysis with x64dbg](lesson_05_x64dbg_dynamic_analysis.md)
   - Launching and attaching to processes
   - Breakpoints, stepping, and register inspection
   - Modifying values at runtime

6. [Lesson 6: Basic Patching and Hooking](lesson_06_patching_and_hooking.md)
   - Changing instructions (branch flips, NOPs)
   - Saving patched binaries
   - Introduction to inline hooks

7. [Lesson 7: DLLs, Imports, and Game Hooking](lesson_07_dlls_and_game_hooks.md)
   - Understanding DLL exports
   - Setting breakpoints on DLL functions
   - Hooking functions in a toy game

8. [Lesson 8: Capstone Project](lesson_08_capstone_patch_your_app.md)
   - Build a Rust/C application with checks and restrictions
   - Fully reverse engineer it
   - Patch it to remove all restrictions
   - Write a short analysis report

---

## How to Use This Course

1. **Read each lesson** from start to finish before attempting exercises
2. **Build all sample binaries** yourself using the provided source code
3. **Attempt every exercise** before looking at solutions
4. **Take notes** on patterns you recognize—you'll see them again
5. **Snapshot your VM** regularly so you can revert if something breaks

---

## Sample Binaries

Throughout this course, you'll work with binaries you compile yourself. This ensures:

- You understand what the code *should* do before you reverse it
- You can compare disassembly to known source code
- No legal or ethical issues with analyzing third-party software

All source code is provided in the lessons. Compile targets using Rust (`rustc`) or C/C++ (MSVC/clang).

---

## Next Steps

Ready to begin? Start with **[Lesson 1: Setting up a Safe Reversing Lab](lesson_01_lab_setup.md)**.

