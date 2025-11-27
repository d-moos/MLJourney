# Binary Reverse Engineering on Windows

A comprehensive three-level course series for software engineers who want to master Windows binary reverse engineering—from foundational concepts to professional-level unpacking and deobfuscation.

---

## Course Overview

This curriculum is designed for **experienced software engineers** (comfortable with Rust/C, basic assembly, and general Windows usage) who want to become proficient—and eventually expert—Windows reverse engineers.

**Tools used throughout:**

- **Disassembler/Decompiler:** Binary Ninja
- **Debugger:** x64dbg
- **PE Utilities:** PE-bear, CFF Explorer
- **System Inspection:** Process Explorer, ProcMon
- **Scripting:** Python

**Target architecture:** Primarily x86-64, with 32-bit x86 mentioned where relevant.

---

## Course Structure

### [Beginner: Foundations & Practical Patching](beginner/README.md)

**Duration:** ~10–15 hours / 3–4 weeks

Set up a safe reversing lab and learn the fundamentals:

- PE format and how Windows loads binaries
- Reading x86-64 assembly patterns
- Static analysis with Binary Ninja
- Dynamic analysis with x64dbg
- Basic patching and function hooking
- Working with DLLs and imports

**Capstone:** Reverse and patch your own Rust/C application.

---

### [Intermediate: Internals, Packers & Automation](intermediate/README.md)

**Duration:** ~10–15 hours / 3–4 weeks

Dig deeper into Windows internals and tackle obfuscated binaries:

- Windows internals relevant to reversing (PEB, TEB, memory layout)
- Exceptions, SEH, and TLS callbacks
- Simple packers and manual unpacking
- Control-flow obfuscation basics
- Memory corruption vulnerabilities
- Toy malware analysis
- Automation with Python (Binary Ninja API, x64dbg scripting)

**Capstone:** Unpack and analyze a custom-packed sample.

---

### [Advanced: Unpacking, Deobfuscation & Professional Reversing](advanced/README.md)

**Duration:** ~10–15 hours / 3–4 weeks

Reach professional-level skills for real-world protected binaries:

- Advanced anti-debugging and anti-VM techniques
- Commercial protectors (VMProtect, Themida) and their internals
- Manual unpacking of VM-based protectors
- Deobfuscation strategies and tooling
- Extracting manually mapped DLLs from memory
- Large application reversing with PDBs and symbol servers
- Building custom unpacking and analysis tools

**Capstone:** Fully unpack and analyze a protected binary.

---

## Prerequisites

Before starting the **Beginner** course, you should:

- Be comfortable programming in at least one systems language (Rust, C, or C++)
- Have a basic idea of assembly concepts (registers, stack, instructions)
- Understand general Windows usage (not necessarily internals)
- Have access to a Windows 10/11 VM with admin rights

---

## How to Use This Course

1. **Work through each level sequentially.** Later lessons assume knowledge from earlier ones.
2. **Do all the exercises.** This is a hands-on discipline—reading alone won't build skills.
3. **Build the sample binaries yourself** (source provided) so you understand what you're reversing.
4. **Use solutions as learning aids**, not shortcuts. Try each exercise before checking the answer.
5. **Take notes** and build your own "reversing cheat sheet" as you learn patterns.

---

## Legal & Ethical Note

All exercises in this course use **binaries you build yourself** or **publicly available crackmes/samples designed for learning**. Never apply these techniques to software you don't own or have explicit permission to analyze.

The skills taught here are intended for:

- Security research and vulnerability analysis
- Malware analysis and incident response
- Software compatibility and interoperability research
- Personal education and CTF competitions

---

## Table of Contents

- [Beginner Course](beginner/README.md)
  - [Lesson 1: Setting up a Safe Reversing Lab](beginner/lesson_01_lab_setup.md)
  - [Lesson 2: x86-64 Assembly Refresher](beginner/lesson_02_assembly_refresher.md)
  - [Lesson 3: PE Format Basics](beginner/lesson_03_pe_format.md)
  - [Lesson 4: Static Analysis with Binary Ninja](beginner/lesson_04_static_analysis.md)
  - [Lesson 5: Dynamic Analysis with x64dbg](beginner/lesson_05_dynamic_analysis.md)
  - [Lesson 6: Basic Patching and Hooking](beginner/lesson_06_patching_hooking.md)
  - [Lesson 7: DLLs, Imports, and Game Hooking](beginner/lesson_07_dlls_imports.md)
  - [Lesson 8: Capstone Project](beginner/lesson_08_capstone.md)

- [Intermediate Course](intermediate/README.md)
  - [Lesson 1: Windows Internals for Reversers](intermediate/lesson_01_windows_internals.md)
  - [Lesson 2: Exceptions, SEH, and TLS Callbacks](intermediate/lesson_02_exceptions_tls.md)
  - [Lesson 3: Simple Packers and Manual Unpacking](intermediate/lesson_03_packers_unpacking.md)
  - [Lesson 4: Control-Flow Obfuscation](intermediate/lesson_04_control_flow_obfuscation.md)
  - [Lesson 5: Vulnerability Patterns](intermediate/lesson_05_vulnerability_patterns.md)
  - [Lesson 6: Toy Malware Analysis](intermediate/lesson_06_malware_analysis.md)
  - [Lesson 7: Automation with Python](intermediate/lesson_07_automation.md)
  - [Lesson 8: Capstone Project](intermediate/lesson_08_capstone.md)

- [Advanced Course](advanced/README.md)
  - [Lesson 1: Advanced Anti-debugging and Anti-VM](advanced/lesson_01_anti_debug_vm.md)
  - [Lesson 2: Commercial Protectors Overview](advanced/lesson_02_commercial_protectors.md)
  - [Lesson 3: Unpacking VM-based Protectors](advanced/lesson_03_vm_unpacking.md)
  - [Lesson 4: Deobfuscation Strategies](advanced/lesson_04_deobfuscation.md)
  - [Lesson 5: Extracting Manually Mapped DLLs](advanced/lesson_05_mapped_dlls.md)
  - [Lesson 6: Large Application Reversing with PDBs](advanced/lesson_06_pdbs_symbols.md)
  - [Lesson 7: Building Custom Unpacking Tools](advanced/lesson_07_custom_tools.md)
  - [Lesson 8: Capstone Project](advanced/lesson_08_capstone.md)

