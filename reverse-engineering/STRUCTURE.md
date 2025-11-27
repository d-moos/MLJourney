# Binary Reverse Engineering on Windows – Course Structure

This document outlines the complete structure of the three-level course series on Windows binary reverse engineering.

---

## Directory Structure

```
reverse-engineering/
├── README.md                          # Main course overview
├── STRUCTURE.md                       # This file
├── beginner/
│   ├── README.md                      # Beginner course introduction
│   ├── lesson_01_lab_setup.md
│   ├── lesson_02_x86_64_refresher.md
│   ├── lesson_03_pe_format_basics.md
│   ├── lesson_04_binary_ninja_static_analysis.md
│   ├── lesson_05_x64dbg_dynamic_analysis.md
│   ├── lesson_06_patching_and_hooking.md
│   ├── lesson_07_dlls_and_game_hooks.md
│   └── lesson_08_capstone_patch_your_app.md
├── intermediate/
│   ├── README.md                      # Intermediate course introduction
│   ├── lesson_01_windows_internals_for_reversers.md
│   ├── lesson_02_exceptions_seh_tls.md
│   ├── lesson_03_simple_packers_manual_unpacking.md
│   ├── lesson_04_control_flow_obfuscation_basics.md
│   ├── lesson_05_vulnerability_patterns.md
│   ├── lesson_06_toy_malware_analysis.md
│   ├── lesson_07_automation_with_python.md
│   └── lesson_08_capstone_unpack_and_analyze.md
└── advanced/
    ├── README.md                      # Advanced course introduction
    ├── lesson_01_anti_debug_vm.md
    ├── lesson_02_commercial_protectors.md
    ├── lesson_03_vm_unpacking.md
    ├── lesson_04_deobfuscation.md
    ├── lesson_05_mapped_dlls.md
    ├── lesson_06_pdbs_symbols.md
    ├── lesson_07_custom_tools.md
    └── lesson_08_capstone.md
```

---

## Course Levels

### Beginner: Foundations & Practical Patching
- **Duration:** 10–15 hours / 3–4 weeks
- **Focus:** Core concepts, lab setup, basic analysis and patching
- **Capstone:** Reverse and patch your own Rust/C application

### Intermediate: Internals, Packers & Automation
- **Duration:** 10–15 hours / 3–4 weeks
- **Focus:** Windows internals, simple packers, malware-like samples, automation
- **Capstone:** Unpack and analyze a custom-packed sample

### Advanced: Unpacking, Deobfuscation & Professional Reversing
- **Duration:** 10–15 hours / 3–4 weeks
- **Focus:** Commercial protectors (VMProtect, Themida), deobfuscation, manually mapped DLLs, professional tools
- **Capstone:** Choose from three tracks (protected binary deep dive, custom unpacking tool, or malware analysis report)

---

## Key Features

- **Explanatory prose style** (not bullet points) for all conceptual content
- **Hands-on exercises** with detailed solutions for every lesson
- **Self-built binaries** (no third-party software analysis except learning samples)
- **Progressive complexity** from fundamentals to professional-level skills
- **Practical focus** with real tools (Binary Ninja, x64dbg, Python)
- **GitBook-compatible** structure with clear README files and table of contents

---

## Tools Used Throughout

| Tool | Lessons | Purpose |
|------|---------|---------|
| **Binary Ninja** | All | Static analysis and decompilation |
| **x64dbg** | All | Dynamic analysis and debugging |
| **PE-bear / CFF Explorer** | Beginner, Intermediate | PE header inspection |
| **Process Explorer** | Intermediate, Advanced | Process and module inspection |
| **ProcMon** | Intermediate, Advanced | System activity monitoring |
| **Python** | Intermediate, Advanced | Scripting and automation |
| **Scylla** | Intermediate, Advanced | IAT reconstruction |
| **Triton / Miasm** | Advanced | Symbolic execution (optional) |

---

## Learning Progression

1. **Beginner** teaches you to read and modify simple binaries
2. **Intermediate** teaches you to defeat basic protections and automate analysis
3. **Advanced** teaches you to tackle professional-grade protections and build tools

Each level assumes completion of the previous level.

---

## How to Use This Course

1. Start with the main [README.md](README.md) for an overview
2. Choose your starting level (Beginner, Intermediate, or Advanced)
3. Read each level's README.md for course-specific information
4. Work through lessons sequentially
5. Complete all exercises before checking solutions
6. Build sample binaries yourself (source provided)
7. Complete the capstone project

---

## GitBook Integration

This course is structured to work with GitBook:

- Each level has a README.md that serves as the introduction
- Lessons are organized in numbered order
- Links between lessons and to the main README are included
- The structure can be directly imported into GitBook's file structure

To use with GitBook:

1. Create a GitBook project
2. Copy the `reverse-engineering` folder into your GitBook workspace
3. Update `SUMMARY.md` to reference the lesson files
4. GitBook will automatically generate navigation and table of contents

---

## Next Steps

- **New to reversing?** Start with [Beginner Course](beginner/README.md)
- **Know the basics?** Start with [Intermediate Course](intermediate/README.md)
- **Experienced reverser?** Start with [Advanced Course](advanced/README.md)

