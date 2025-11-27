# Getting Started with Binary Reverse Engineering on Windows

Welcome! This guide will help you get started with the course.

---

## 🚀 Quick Start (5 minutes)

1. **Read the main overview:** [README.md](README.md)
2. **Choose your level:**
   - **Beginner** – You're new to reversing
   - **Intermediate** – You know the basics
   - **Advanced** – You want professional skills
3. **Go to your level's README:**
   - [Beginner](beginner/README.md)
   - [Intermediate](intermediate/README.md)
   - [Advanced](advanced/README.md)
4. **Start with Lesson 1**

---

## 📚 Course Overview

This is a **three-level course series** on Windows binary reverse engineering:

- **Beginner (8 lessons):** Learn fundamentals, set up tools, analyze and patch simple binaries
- **Intermediate (8 lessons):** Defeat basic protections, unpack binaries, automate analysis
- **Advanced (8 lessons):** Tackle commercial protectors, deobfuscate code, build custom tools

Each level takes **10–15 hours** and includes **8 lessons with exercises and solutions**.

---

## ✅ Prerequisites

Before starting, you should have:

- **Windows 10/11 VM** with admin rights
- **Rust or C/C++ compiler** (to build sample binaries)
- **Basic assembly knowledge** (registers, stack, instructions)
- **Comfortable with Rust/C programming**
- **General Windows knowledge** (not deep internals)

---

## 🛠️ Tools You'll Need

Install these in your VM:

1. **Binary Ninja** – Disassembler/decompiler (free or paid)
2. **x64dbg** – Debugger (free)
3. **PE-bear** or **CFF Explorer** – PE inspection (free)
4. **Process Explorer** – Process monitoring (free)
5. **ProcMon** – System monitoring (free)
6. **Python 3** – Scripting (free)

All are free or have free versions. See Lesson 1 of your chosen level for installation instructions.

---

## 📖 How to Use This Course

### For Self-Study

1. **Choose your level** (Beginner, Intermediate, or Advanced)
2. **Read the course introduction** for your level
3. **Work through lessons sequentially** – each builds on the previous
4. **Do all exercises** before checking solutions
5. **Build sample binaries yourself** (source code provided)
6. **Complete the capstone project**

### For Team Training

- Use as a structured curriculum for security team onboarding
- Adapt capstone projects to your organization's needs
- Customize tool choices if needed

### For GitBook

- Copy the `reverse-engineering` folder to your GitBook workspace
- Create a `SUMMARY.md` file referencing the lessons
- GitBook will auto-generate navigation

---

## 🎯 What You'll Learn

### Beginner Level
- Set up a safe reversing lab
- Understand PE format (EXE/DLL structure)
- Read x86-64 assembly
- Use Binary Ninja for static analysis
- Use x64dbg for dynamic analysis
- Patch and hook binaries

### Intermediate Level
- Understand Windows internals (PEB, TEB, memory layout)
- Manually unpack simple packed binaries
- Recognize and defeat control-flow obfuscation
- Identify vulnerability patterns
- Analyze malware-like samples
- Automate analysis with Python

### Advanced Level
- Defeat advanced anti-debugging and anti-VM tricks
- Understand commercial protectors (VMProtect, Themida)
- Unpack VM-based protectors
- Deobfuscate complex code
- Extract manually mapped DLLs
- Reverse large applications with PDBs
- Build custom unpacking tools

---

## 📋 File Structure

```
reverse-engineering/
├── README.md                    ← Main overview (start here)
├── INDEX.md                     ← Complete file index
├── GETTING_STARTED.md           ← This file
├── STRUCTURE.md                 ← GitBook integration guide
├── COMPLETION_SUMMARY.md        ← Project statistics
├── beginner/
│   ├── README.md                ← Beginner course intro
│   └── lesson_*.md              ← 8 lessons
├── intermediate/
│   ├── README.md                ← Intermediate course intro
│   └── lesson_*.md              ← 8 lessons
└── advanced/
    ├── README.md                ← Advanced course intro
    └── lesson_*.md              ← 8 lessons
```

---

## 🎓 Learning Path

```
START HERE
    ↓
Read README.md
    ↓
Choose your level
    ↓
Read level's README.md
    ↓
Work through lessons 1-7
    ↓
Complete capstone (lesson 8)
    ↓
EXPERT LEVEL SKILLS
```

---

## 💡 Tips for Success

1. **Build binaries yourself** – Don't skip this. Understanding what you're reversing is crucial.
2. **Do all exercises** – This is where learning happens. Don't just read solutions.
3. **Take notes** – Build your own "reversing cheat sheet" as you learn patterns.
4. **Snapshot your VM** – Regularly save snapshots so you can revert if something breaks.
5. **Be patient** – Reversing is a skill that improves with practice. Some exercises will be challenging.
6. **Document your findings** – Write down what you learn. This builds a reference for future projects.

---

## ❓ FAQ

**Q: Do I need to know assembly?**  
A: You should have a basic idea (registers, stack, instructions). Lesson 2 of Beginner refreshes this.

**Q: Can I skip levels?**  
A: Not recommended. Each level builds on the previous. If you're experienced, you can move faster through Beginner.

**Q: How long does this take?**  
A: ~10–15 hours per level, so 30–45 hours total. Depends on your pace and depth.

**Q: Can I use different tools?**  
A: Yes, but the course is written for Binary Ninja and x64dbg. Adapt as needed.

**Q: Is this legal?**  
A: Yes. All exercises use binaries you build yourself. Never apply these techniques to software you don't own.

---

## 🔗 Navigation

- **Main Course:** [README.md](README.md)
- **Complete Index:** [INDEX.md](INDEX.md)
- **Beginner Course:** [beginner/README.md](beginner/README.md)
- **Intermediate Course:** [intermediate/README.md](intermediate/README.md)
- **Advanced Course:** [advanced/README.md](advanced/README.md)

---

## 🚀 Ready to Start?

1. **Beginner?** → Go to [beginner/README.md](beginner/README.md)
2. **Intermediate?** → Go to [intermediate/README.md](intermediate/README.md)
3. **Advanced?** → Go to [advanced/README.md](advanced/README.md)

**Let's begin!**

