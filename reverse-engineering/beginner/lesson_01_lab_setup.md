# Lesson 1: Setting up a Safe Reversing Lab

## Overview

Before you can begin reverse engineering, you need a safe, isolated environment where you can experiment without risk. In this lesson, you will build a **reproducible Windows reversing lab** inside a virtual machine, install all the core tools, and compile your first target binary. This environment will be your sandbox for the entire course.

The key principle is **isolation**: by working in a VM, you can safely run untrusted binaries, make mistakes, and revert to a clean state. You'll also learn why snapshots are essential for reversing work—they let you save your progress and roll back if something breaks.

## What You'll Learn

By the end of this lesson, you will understand:

- **Why virtual machines are essential** for reverse engineering (isolation, snapshots, reproducibility)
- **How to set up a Windows 10/11 VM** with appropriate resources
- **How to install all required tools** (Binary Ninja, x64dbg, PE-bear, Process Explorer, ProcMon, Python, Rust)
- **How to create and use snapshots** to save and restore your lab state
- **How to compile a simple Rust program** into a Windows x86-64 executable
- **How to verify your setup** by inspecting a compiled binary

## Prerequisites

Before starting this lesson, you should have:

- Access to a host machine capable of running a Windows VM (Windows, macOS, or Linux)
- At least 50 GB of free disk space for the VM
- At least 8 GB of RAM available (16 GB recommended)
- Basic familiarity with installing software on Windows
- Comfortable using a terminal and a text editor

## Why Use a Virtual Machine?

Reverse engineering often involves running untrusted or potentially malicious code. Even if you're only analyzing binaries you compiled yourself, mistakes happen. A virtual machine provides several critical benefits:

**Isolation**: Your VM is completely separate from your host machine. If something goes wrong—a crash, a malware infection, or accidental system modification—your host is unaffected.

**Snapshots**: Most hypervisors let you save the entire state of a VM (memory, disk, everything) at a specific point in time. If you make a mistake or want to try something risky, you can revert to a previous snapshot instantly. This is invaluable for reversing work.

**Reproducibility**: You can create multiple snapshots at different stages (clean install, tools installed, first binary compiled, etc.). This lets you quickly reset to a known state or compare different configurations.

**Experimentation**: You can freely modify files, install software, change settings, and run untrusted binaries without worrying about breaking your system.

For these reasons, **always work in a VM** when reverse engineering. Never reverse engineer on your main machine.

## Choosing a Hypervisor

A hypervisor is software that lets you run virtual machines. Here are the most common options:

**VMware Workstation Pro** (Windows/Linux) or **VMware Fusion** (macOS) are industry-standard hypervisors with excellent performance and snapshot support. They're commercial but widely used in security work.

**VirtualBox** (Windows/macOS/Linux) is free and open-source. It's less performant than VMware but perfectly adequate for this course and has good snapshot support.

**Hyper-V** (Windows) is built into Windows Pro/Enterprise editions. It's free if you have the right Windows version and has decent performance.

**Parallels Desktop** (macOS) is commercial but popular on macOS and has good performance.

For this course, **VirtualBox is recommended** because it's free, cross-platform, and has all the features you need. If you already have VMware or Hyper-V, those work fine too.

## Creating a Windows 10/11 VM

### Step 1: Download Windows

Download a Windows 10 or Windows 11 ISO from Microsoft:

- **Windows 10**: https://www.microsoft.com/en-us/software-download/windows10
- **Windows 11**: https://www.microsoft.com/en-us/software-download/windows11

You'll need a valid license to activate Windows, but you can use it unactivated for 30 days (sufficient for this course).

### Step 2: Create the VM

Using your hypervisor (VirtualBox, VMware, etc.):

1. Create a new virtual machine
2. Allocate **at least 4 CPU cores** (8 recommended)
3. Allocate **at least 8 GB of RAM** (16 GB recommended)
4. Create a **50 GB virtual disk** (dynamically allocated is fine)
5. Attach the Windows ISO as the boot drive
6. Boot the VM and follow the Windows installation wizard

### Step 3: Install Guest Additions/Tools

After Windows is installed, install the hypervisor's guest additions (this improves performance and enables features like clipboard sharing):

- **VirtualBox**: Insert Guest Additions CD (usually automatic) or download from VirtualBox menu
- **VMware**: Install VMware Tools from the VM menu
- **Hyper-V**: Install Hyper-V Integration Services

Reboot after installation.

### Step 4: Update Windows

Open Windows Update and install all available updates. This may take a while and require multiple reboots. Wait for it to complete.

## Installing Required Tools

Now that you have a clean Windows VM, install the reversing tools. You can use the provided PowerShell script (`install_tools.ps1`) or install manually.

### Using the Installation Script (Recommended)

1. Copy `install_tools.ps1` to your VM
2. Open PowerShell as Administrator
3. Run: `powershell -ExecutionPolicy Bypass -File install_tools.ps1`
4. Wait for installation to complete
5. Install Binary Ninja manually from https://binary.ninja/

### Manual Installation

If the script fails or you prefer manual installation:

1. **Install Chocolatey** (package manager):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
   [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
   iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. **Install tools via Chocolatey**:
   ```powershell
   choco install x64dbg python rustup.install pe-bear procexp procmon -y
   ```

3. **Install Binary Ninja** manually from https://binary.ninja/

4. **Verify installations**:
   ```powershell
   x64dbg --version
   python --version
   rustc --version
   ```

## Creating Your First Snapshot

Once all tools are installed and verified, create a snapshot of your clean lab:

1. In your hypervisor, select "Take Snapshot" or "Create Snapshot"
2. Name it `clean-reversing-lab`
3. Add a description: "Clean Windows VM with all reversing tools installed"
4. Save the snapshot

This snapshot is your "reset button". If you ever break something or want to start fresh, you can revert to this snapshot and be back to a clean state in seconds.

## Installing Rust

Rust is used to compile sample binaries for this course. Install it:

1. Download the Rust installer from https://www.rust-lang.org/tools/install
2. Run the installer and follow the prompts
3. Accept the default installation (MSVC toolchain)
4. Verify installation:
   ```powershell
   rustc --version
   cargo --version
   ```

If you prefer C/C++, you can install Visual Studio Community instead, but Rust is recommended for this course.

## Compiling Your First Binary

Now let's compile a simple Rust program to verify everything works.

### Create a Rust Project

```powershell
cargo new --bin hello_reversing
cd hello_reversing
```

### Write the Program

Edit `src/main.rs`:

```rust
use std::io::{self, Write};

fn main() {
    println!("Welcome to Binary Reversing!");
    println!("Enter your name: ");

    let mut name = String::new();
    io::stdin().read_line(&mut name).expect("Failed to read line");

    let name = name.trim();

    if name.is_empty() {
        println!("You didn't enter a name!");
    } else {
        println!("Hello, {}!", name);
        println!("Your name has {} characters.", name.len());
    }
}
```

### Compile the Binary

```powershell
cargo build --release
```

The compiled binary is at: `target/release/hello_reversing.exe`

### Test the Binary

```powershell
.\target\release\hello_reversing.exe
```

Enter your name and verify it works correctly.

## Inspecting Your Binary

Now open your compiled binary in PE-bear to see its structure:

1. Open PE-bear
2. File → Open → select `hello_reversing.exe`
3. Explore the PE structure:
   - **DOS Header**: Shows the MZ signature and PE offset
   - **PE Header**: Shows machine type (x64), number of sections, etc.
   - **Sections**: Lists `.text` (code), `.data` (data), `.rdata` (read-only data), etc.
   - **Imports**: Shows which DLLs are imported (kernel32.dll, etc.)
   - **Entry Point**: Shows the RVA where execution starts

Take note of these values—you'll see them again throughout the course.

## Exercises

### Exercise 1: Create the VM and Snapshot

**Objective**: Set up a complete reversing lab with all tools installed.

**Steps**:
1. Create a Windows 10/11 VM with at least 4 CPU cores and 8 GB RAM
2. Install Windows and all updates
3. Run the installation script (or install tools manually)
4. Verify all tools are installed:
   - Open x64dbg and confirm it launches
   - Run `python --version` in PowerShell
   - Run `rustc --version` in PowerShell
   - Open PE-bear and confirm it launches
   - Open Process Explorer and confirm it launches
5. Create a snapshot named `clean-reversing-lab`

**Verification**: You should have a working VM with all tools installed and a snapshot saved.

### Exercise 2: Compile a Rust Test Program

**Objective**: Create and compile a simple Rust program.

**Steps**:
1. Create a new Rust project: `cargo new --bin my_first_binary`
2. Write a program that:
   - Prints a welcome message
   - Reads input from the user
   - Performs some simple calculation (e.g., count characters, reverse string)
   - Prints the result
3. Compile it: `cargo build --release`
4. Run it and verify it works correctly
5. Note the path to the compiled `.exe` file

**Verification**: You should have a working `.exe` file that runs correctly.

### Exercise 3: Inspect the Binary in PE-bear

**Objective**: Understand the structure of a compiled Windows binary.

**Steps**:
1. Open your compiled `.exe` in PE-bear
2. Identify and document:
   - **Machine type**: Should be "x64" (AMD64)
   - **Number of sections**: Usually 4-6 sections
   - **Entry point RVA**: The address where execution starts
   - **Imported DLLs**: List all DLLs imported by your binary
   - **Imported functions**: List at least 5 functions imported from kernel32.dll
3. Create a text file documenting these findings

**Verification**: You should understand the basic structure of a PE file and be able to identify key components.

## Solutions

### Solution 1: VM and Snapshot Setup

After following the steps above, you should have:

1. A Windows 10/11 VM running with:
   - 4+ CPU cores
   - 8+ GB RAM
   - All Windows updates installed
   - All reversing tools installed and verified

2. A snapshot named `clean-reversing-lab` that you can revert to at any time

**Verification commands**:
```powershell
# Check Python
python --version
# Output: Python 3.x.x

# Check Rust
rustc --version
# Output: rustc 1.x.x (...)

# Check x64dbg
x64dbg --version
# Output: x64dbg version ...
```

### Solution 2: Compiled Rust Program

Your compiled binary should:
- Be located at `target/release/my_first_binary.exe`
- Run without errors
- Accept user input
- Produce output based on that input

Example output:
```
Welcome to Binary Reversing!
Enter your name:
Alice
Hello, Alice!
Your name has 5 characters.
```

### Solution 3: PE-bear Analysis

When you open your binary in PE-bear, you should see:

**DOS Header**:
- Magic: `MZ` (0x5A4D)
- PE offset: Usually 0x40 or similar

**PE Header**:
- Machine: `0x8664` (x64)
- Number of sections: 4-6
- Characteristics: `0x0022` (executable, large address aware)

**Sections** (typical for Rust binaries):
- `.text`: Code section (executable)
- `.data`: Initialized data
- `.rdata`: Read-only data (strings, constants)
- `.reloc`: Relocation information
- `.rsrc`: Resources (if any)

**Imports** (typical for a simple Rust program):
- `kernel32.dll`: Core Windows API
  - `GetStdHandle`: Get standard input/output handles
  - `WriteFile`: Write to console
  - `ReadFile`: Read from console
  - `ExitProcess`: Exit the program
- `ntdll.dll`: Native API (used internally)

**Entry Point**:
- RVA: Usually 0x1000 or similar (start of `.text` section)

## Summary

You now have a complete reversing lab set up! You've learned:

- Why virtual machines are essential for safe reversing
- How to create and configure a Windows VM
- How to install all required tools
- How to use snapshots to save and restore your lab state
- How to compile a simple Rust program
- How to inspect a compiled binary's structure

In the next lesson, you'll dive deeper into x86-64 assembly and learn to read the code inside these binaries.

- Concrete installation steps and screenshots
- Example Rust source code and compilation commands
- Example screenshots from PE-bear highlighting sections, entry point, and imports

