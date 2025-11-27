# Intermediate Course: Internals, Packers & Automation

Welcome to the **Intermediate** course on Windows binary reverse engineering. This course builds on the Beginner foundations to tackle more complex binaries, including packed and lightly obfuscated samples.

---

## Who This Course Is For

You have completed the Beginner course (or equivalent) and can:

- Navigate PE files and understand their structure
- Read basic x86-64 assembly and recognize common patterns
- Use Binary Ninja for static analysis and x64dbg for debugging
- Perform simple patches and understand inline hooking concepts

Now you want to go deeper: understand Windows internals, defeat basic protections, analyze malware-like samples, and automate your workflows.

---

## What You'll Learn

By the end of this course, you will be able to:

1. **Navigate Windows internals** relevant to reversing (PEB, TEB, memory layout, modules)
2. **Understand exception handling** (SEH) and **TLS callbacks**
3. **Manually unpack** simple packed binaries and reconstruct imports
4. **Recognize and reason about** control-flow obfuscation
5. **Identify vulnerability patterns** (stack overflow, use-after-free) in binaries
6. **Analyze toy malware** samples for behavior, persistence, and indicators
7. **Automate analysis** using Python with Binary Ninja's API and x64dbg scripting
8. **Complete a full unpack-and-analyze workflow** on a custom-packed sample

---

## Prerequisites

Before starting, you should:

- Have completed the **Beginner course** or have equivalent skills
- Have your reversing VM set up with all tools from the Beginner course
- Be comfortable with Python basics (for automation lessons)

---

## Tools (Additions for This Course)

In addition to the Beginner toolset, you'll use:

| Tool | Purpose |
|------|---------|
| **Scylla** | IAT reconstruction for unpacked binaries |
| **x64dbg Python plugin** or **scripting** | Automated debugging workflows |
| **Binary Ninja Python API** | Scripted static analysis |

---

## Course Duration

**Estimated time:** 10–15 hours over 3–4 weeks

Exercises are more involved than the Beginner course. Budget extra time for the unpacking and automation lessons.

---

## Table of Contents

1. [Lesson 1: Windows Internals for Reversers](lesson_01_windows_internals_for_reversers.md)
   - Process and thread structures (PEB, TEB)
   - Virtual memory layout and module lists
   - Heap basics and why they matter

2. [Lesson 2: Exceptions, SEH, and TLS Callbacks](lesson_02_exceptions_seh_tls.md)
   - Structured Exception Handling internals
   - Recognizing SEH in disassembly
   - TLS callbacks and their use in protections

3. [Lesson 3: Simple Packers and Manual Unpacking](lesson_03_simple_packers_manual_unpacking.md)
   - How packers work (stub + payload)
   - Finding the Original Entry Point (OEP)
   - Dumping and rebuilding unpacked binaries

4. [Lesson 4: Control-Flow Obfuscation](lesson_04_control_flow_obfuscation_basics.md)
   - Opaque predicates and junk code
   - Control-flow flattening basics
   - Manual deobfuscation strategies

5. [Lesson 5: Vulnerability Patterns](lesson_05_vulnerability_patterns.md)
   - Stack buffer overflows in practice
   - Use-after-free recognition
   - Mapping vulnerabilities in disassembly

6. [Lesson 6: Toy Malware Analysis](lesson_06_toy_malware_analysis.md)
   - Behavioral analysis workflow
   - Identifying persistence, file drops, registry changes
   - Writing a basic analysis report

7. [Lesson 7: Automation with Python](lesson_07_automation_with_python.md)
   - Binary Ninja scripting fundamentals
   - x64dbg scripting and plugins
   - Building reusable analysis tools

8. [Lesson 8: Capstone Project](lesson_08_capstone_unpack_and_analyze.md)
   - Custom-packed sample (provided or self-built)
   - Full manual unpacking workflow
   - Post-unpack analysis and documentation

---

## Sample Binaries

This course uses binaries you build yourself, plus some you'll pack with a simple custom packer. All source code and packer code is provided.

For the malware analysis lesson, you'll work with synthetic "malware-like" samples that simulate real behaviors (file drops, registry writes, process spawning) without any actual malicious capability.

---

## How to Use This Course

1. **Review Beginner concepts** if any feel rusty—this course assumes that foundation
2. **Work in your VM** with snapshots; some exercises involve binaries that modify system state
3. **Take your time with unpacking**—it's a skill that improves with practice
4. **Build your own tools** in the automation lesson; these will serve you in the Advanced course

---

## Next Steps

Ready to continue? Start with **[Lesson 1: Windows Internals for Reversers](lesson_01_windows_internals_for_reversers.md)**.

