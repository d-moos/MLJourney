# Advanced Course: Unpacking, Deobfuscation & Professional Reversing

Welcome to the **Advanced** course on Windows binary reverse engineering. This course focuses on the skills needed to tackle professionally protected binaries—commercial packers, VM-based obfuscators, and real-world applications with symbols.

---

## Who This Course Is For

You have completed the Beginner and Intermediate courses (or equivalent) and can:

- Manually unpack simple packed binaries
- Recognize and work around basic control-flow obfuscation
- Automate analysis tasks with Python
- Analyze malware-like samples and write reports

Now you want to reach **professional-level** skills: defeating commercial protections like VMProtect and Themida, extracting hidden modules from memory, and efficiently reversing large applications.

---

## What You'll Learn

By the end of this course, you will be able to:

1. **Defeat advanced anti-debugging and anti-VM** techniques used by commercial protectors
2. **Understand how commercial protectors work** (VMProtect, Themida, and similar)
3. **Unpack VM-based protectors** by tracing virtualized code and extracting original logic
4. **Apply systematic deobfuscation strategies** including symbolic execution concepts
5. **Extract manually mapped DLLs** from process memory (common in game cheats, malware, and loaders)
6. **Reverse large applications** using PDBs, symbol servers, and efficient navigation
7. **Build custom unpacking and analysis tools** in Python/Rust
8. **Complete professional-level reversing tasks** on protected binaries

---

## Prerequisites

Before starting, you must:

- Have completed both **Beginner and Intermediate courses** (or have equivalent experience)
- Be comfortable with:
  - Manual unpacking and IAT reconstruction
  - Python scripting with Binary Ninja API
  - Windows internals (PEB, TEB, memory layout, SEH)
- Have significant patience—advanced protections are designed to waste your time

---

## Tools (Additions for This Course)

| Tool | Purpose |
|------|---------|
| **VMProtect/Themida samples** | Self-built protected test binaries |
| **Triton** or **Miasm** (optional) | Symbolic execution for deobfuscation |
| **Custom Python/Rust tools** | You'll build these during the course |
| **WinDbg** (optional) | Kernel debugging and advanced scenarios |

---

## Course Duration

**Estimated time:** 10–15 hours over 3–4 weeks

These lessons are dense. Some exercises (especially VM unpacking) may take multiple sessions. That's normal—professional reversing is slow and methodical.

---

## Table of Contents

1. [Lesson 1: Advanced Anti-debugging and Anti-VM Techniques](lesson_01_anti_debug_vm.md)
   - Comprehensive anti-debug tricks and bypasses
   - VM/sandbox detection techniques
   - Building a systematic bypass toolkit

2. [Lesson 2: Commercial Protectors Overview](lesson_02_commercial_protectors.md)
   - How VMProtect, Themida, and similar tools work
   - Protection layers: packing, virtualization, mutation
   - Identifying which protector was used

3. [Lesson 3: Unpacking VM-based Protectors](lesson_03_vm_unpacking.md)
   - Understanding bytecode virtualization
   - Tracing VM handlers and bytecode
   - Extracting original logic from virtualized code

4. [Lesson 4: Deobfuscation Strategies](lesson_04_deobfuscation.md)
   - Pattern-based deobfuscation
   - Symbolic execution concepts
   - Building deobfuscation scripts

5. [Lesson 5: Extracting Manually Mapped DLLs](lesson_05_mapped_dlls.md)
   - How manual mapping works (no LoadLibrary)
   - Detecting hidden modules in memory
   - Dumping and reconstructing mapped DLLs

6. [Lesson 6: Large Application Reversing with PDBs](lesson_06_pdbs_symbols.md)
   - Configuring symbol servers
   - Navigating large codebases efficiently
   - Combining public symbols with reverse engineering

7. [Lesson 7: Building Custom Unpacking Tools](lesson_07_custom_tools.md)
   - Designing reusable unpacking frameworks
   - Automating OEP detection and dumping
   - Import reconstruction automation

8. [Lesson 8: Capstone Project](lesson_08_capstone.md)
   - Protected binary (VMProtect or custom VM)
   - Full analysis: unpack, deobfuscate, document
   - Professional-quality deliverable

---

## Sample Binaries

You will create your own protected samples by:

1. Writing simple Rust/C programs
2. Protecting them with VMProtect trial (or a custom VM you build)
3. Then reversing your own protected code

This ensures you understand both sides: how protections are applied and how to defeat them.

---

## A Note on Commercial Protectors

VMProtect and Themida offer trial versions suitable for learning. You will:

- Only protect binaries **you wrote yourself**
- Never distribute protected cracks or bypasses
- Focus on understanding the *techniques*, which transfer to many protection schemes

The skills you learn apply broadly: game anti-cheats, malware packers, and proprietary protection all use similar concepts.

---

## How to Use This Course

1. **Expect to struggle**—these protections are designed by experts to resist analysis
2. **Document everything**—patterns you discover will help in future projects
3. **Build tools, not just knowledge**—automation is essential at this level
4. **Join communities** (Tuts4You, RE Discord servers, etc.) to learn from others

---

## Next Steps

Ready for the challenge? Start with **[Lesson 1: Advanced Anti-debugging and Anti-VM Techniques](lesson_01_anti_debug_vm.md)**.

