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

A hypervisor is software that lets you run virtual machines. Selecting the right hypervisor is important because you'll be spending significant time working within this environment, and different hypervisors offer varying levels of performance, features, and ease of use.

**VMware Workstation Pro** (available for Windows and Linux) and **VMware Fusion** (for macOS) represent the industry standard for professional virtualization work. These commercial products offer excellent performance, robust snapshot support, and advanced features like network simulation and USB device passthrough. They're widely used in security research and penetration testing environments, which means you'll find extensive documentation and community support. The main drawback is cost, though educational licenses are sometimes available.

**VirtualBox** (available for Windows, macOS, and Linux) is a free and open-source alternative that has become extremely popular in the security community. While it doesn't match VMware's raw performance, it's more than adequate for reverse engineering work and offers solid snapshot functionality. The cross-platform nature means you can move your VM between different host operating systems if needed, and the active community provides excellent plugin support.

**Hyper-V** (Windows only) comes built into Windows Pro and Enterprise editions, making it a zero-cost option if you're already running these Windows versions. It offers good performance and integrates tightly with Windows, but it's less flexible than VMware or VirtualBox for certain reverse engineering scenarios. One notable limitation is that Hyper-V can conflict with other hypervisors, so you may need to choose between Hyper-V and VirtualBox/VMware on the same machine.

**Parallels Desktop** (macOS only) is another commercial option that's particularly popular among Mac users. It offers excellent performance and seamless macOS integration, making it a good choice if you're working primarily on Apple hardware.

For this course, **VirtualBox is the recommended choice** because it's completely free, works across all major operating systems, and provides all the features you'll need for reverse engineering work. The snapshot system is reliable, the interface is intuitive, and the community support is excellent. If you already have VMware Workstation, Fusion, or Hyper-V set up and are comfortable with it, those will work perfectly fine as well—the core concepts remain the same across all hypervisors.

## Creating a Windows 10/11 VM

### Step 1: Download Windows

Download a Windows 10 or Windows 11 ISO from Microsoft:

- **Windows 10**: https://www.microsoft.com/en-us/software-download/windows10
- **Windows 11**: https://www.microsoft.com/en-us/software-download/windows11

You'll need a valid license to activate Windows, but you can use it unactivated for 30 days (sufficient for this course).

### Step 2: Create the VM

Using your chosen hypervisor (VirtualBox, VMware, etc.), you'll now create a new virtual machine with appropriate resources for reverse engineering work. The resource allocation is important—too little and your tools will run slowly, too much and you'll starve your host system.

Start by creating a new virtual machine in your hypervisor's interface. When prompted for the operating system type, select Windows 10 or Windows 11 (64-bit). For CPU allocation, assign **at least 4 CPU cores**, though 8 cores is recommended if your host system can spare them. Reverse engineering tools like Binary Ninja and debuggers can be CPU-intensive, especially when analyzing large binaries or performing automated analysis.

For memory allocation, assign **at least 8 GB of RAM**, with 16 GB being the recommended amount. This ensures smooth operation when you have multiple tools open simultaneously—a common scenario when you're comparing static analysis in Binary Ninja with dynamic analysis in x64dbg while monitoring system calls in Process Monitor.

Create a **50 GB virtual disk** for the VM. You can use dynamically allocated storage, which means the disk file will only grow as you use space rather than immediately consuming 50 GB on your host. This is sufficient for Windows, all tools, and a reasonable collection of binaries to analyze.

Attach the Windows ISO you downloaded as the virtual CD/DVD drive and set it as the boot device. Start the VM and follow the Windows installation wizard. You can skip entering a product key during installation—Windows will run in evaluation mode for 30 days, which is more than enough time to complete this course.

### Step 3: Install Guest Additions/Tools

After Windows installation completes, the next critical step is installing your hypervisor's guest additions or tools. These are special drivers and utilities that dramatically improve VM performance and enable convenient features that make working in a VM much more pleasant.

**For VirtualBox users**: After logging into Windows, go to the VirtualBox menu and select "Devices" → "Insert Guest Additions CD image". This mounts a virtual CD containing the installer. Open File Explorer, navigate to the CD drive, and run the installer. The Guest Additions enable features like automatic screen resolution adjustment, seamless mouse integration (no more clicking to release the mouse), shared clipboard between host and VM, and drag-and-drop file transfer.

**For VMware users**: Select "VM" → "Install VMware Tools" from the menu. This mounts the VMware Tools installer, which you can run from the virtual CD drive. VMware Tools provides similar benefits to VirtualBox Guest Additions, along with improved graphics performance and time synchronization.

**For Hyper-V users**: Hyper-V Integration Services are typically installed automatically on modern Windows versions. You can verify they're running by checking Services (services.msc) for Hyper-V services. These provide basic integration features, though they're less feature-rich than VMware or VirtualBox equivalents.

After installation completes, reboot the VM to ensure all drivers and services are properly loaded.

### Step 4: Update Windows

Before installing any reverse engineering tools, it's essential to fully update Windows. Open the Settings app, navigate to "Windows Update", and click "Check for updates". Windows will download and install all available updates, which may include security patches, driver updates, and feature improvements.

This process can take considerable time—sometimes 30 minutes to an hour or more—and will likely require multiple reboots as different update batches are applied. Be patient and let it complete fully. You'll know you're done when Windows Update reports "You're up to date" with no pending updates. Having a fully updated system ensures compatibility with modern reverse engineering tools and prevents potential issues caused by outdated system libraries.

## Installing Required Tools

Now that you have a clean Windows VM, install the reversing tools. You can use the provided PowerShell script (`install_tools.ps1`) or install manually.

### Using the Installation Script (Recommended)

The quickest way to get your lab environment set up is to use the provided PowerShell installation script, which automates the installation of most tools. First, copy the `install_tools.ps1` file from the course materials to your VM (you can use the shared clipboard feature enabled by Guest Additions/Tools, or download it directly if you have network access).

Next, open PowerShell with Administrator privileges—this is essential because the script needs elevated permissions to install software. You can do this by right-clicking the Start menu and selecting "Windows PowerShell (Admin)" or "Terminal (Admin)" on Windows 11.

Once PowerShell is open, navigate to the directory containing the script and run it with: `powershell -ExecutionPolicy Bypass -File install_tools.ps1`. The `-ExecutionPolicy Bypass` flag is necessary because Windows blocks unsigned scripts by default for security reasons. The script will download and install x64dbg, Python, Rust, PE-bear, Process Explorer, and Process Monitor automatically.

Wait for the installation to complete—this typically takes 5-10 minutes depending on your internet connection. The script will display progress messages as it installs each tool. Note that Binary Ninja cannot be installed automatically due to licensing requirements, so you'll need to download and install it manually from https://binary.ninja/ after the script completes. Binary Ninja offers a free demo version that's sufficient for learning, though you may want to consider the commercial or educational license for full features.

### Manual Installation

If the automated script fails due to network issues, permission problems, or if you simply prefer to understand exactly what's being installed, you can perform a manual installation. This approach also gives you more control over which versions of each tool you install.

**First, install Chocolatey**, which is a package manager for Windows similar to apt on Linux or Homebrew on macOS. Chocolatey makes it much easier to install and update command-line tools. Open PowerShell as Administrator and run the following commands:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

The first line allows the execution of signed scripts for this PowerShell session. The second line ensures that PowerShell uses modern TLS protocols for secure downloads. The third line downloads and executes the Chocolatey installer.

**Next, install the core tools via Chocolatey**. Once Chocolatey is installed, you can install multiple tools with a single command:

```powershell
choco install x64dbg python rustup.install pe-bear procexp procmon -y
```

The `-y` flag automatically confirms all prompts, making the installation non-interactive. This command installs x64dbg (our primary debugger), Python (for automation scripts), Rust (for compiling sample binaries), PE-bear (a PE file viewer), Process Explorer (advanced task manager), and Process Monitor (system call monitor).

**Install Binary Ninja manually** by downloading it from https://binary.ninja/. Run the installer and follow the prompts. You'll need to create an account and activate your license (or use the demo mode).

**Finally, verify all installations** by checking the version of each tool. This confirms that the tools are properly installed and accessible from the command line:

```powershell
x64dbg --version
python --version
rustc --version
```

If any of these commands fail, the tool may not be in your PATH, or the installation may have encountered an error. You may need to restart PowerShell or your entire VM for PATH changes to take effect.

## Creating Your First Snapshot

Once all tools are installed and verified to be working correctly, it's time to create your first snapshot. This is one of the most important steps in setting up your reversing lab, as it creates a restore point you can return to at any time.

In your hypervisor's menu, look for the snapshot functionality—in VirtualBox, this is under "Machine" → "Take Snapshot"; in VMware, it's "VM" → "Snapshot" → "Take Snapshot". When prompted, name your snapshot something descriptive like `clean-reversing-lab`. Add a detailed description such as "Clean Windows VM with all reversing tools installed and verified - created [date]". Including the date helps you track when the snapshot was created, which can be useful if you create multiple snapshots over time.

Save the snapshot and wait for the process to complete. Depending on your VM's current memory usage and disk size, this might take anywhere from a few seconds to a couple of minutes. Once complete, this snapshot becomes your "reset button"—a safety net that lets you experiment freely. If you accidentally break something, install malware that won't clean up properly, or simply want to start fresh, you can revert to this snapshot and be back to a pristine state in seconds. This is invaluable for reverse engineering work where you'll often be running untrusted or modified binaries.

## Installing Rust

Rust is the primary language used to compile sample binaries throughout this course. While you could use C or C++ instead, Rust is recommended because it produces clean, modern binaries with predictable structure, and the compiler provides excellent error messages that help you understand what's happening at the binary level.

If Rust wasn't installed via the Chocolatey script above, you'll need to install it manually. Download the Rust installer (rustup-init.exe) from https://www.rust-lang.org/tools/install. Run the installer, which will present you with installation options. For this course, accept the default installation, which uses the MSVC (Microsoft Visual C++) toolchain. This is important because it ensures your compiled binaries use the standard Windows ABI and calling conventions, making them representative of real-world Windows applications.

The installer will download and configure the Rust compiler (rustc), the Cargo build system and package manager, and the standard library. This process typically takes a few minutes. Once installation completes, you may need to restart your PowerShell session for the PATH changes to take effect.

Verify the installation by checking the versions of both the compiler and build system:

```powershell
rustc --version
cargo --version
```

Both commands should display version information. If they do, Rust is properly installed and ready to use. If you encounter "command not found" errors, try restarting PowerShell or your entire VM to ensure PATH changes are applied.

If you prefer C/C++ and are already familiar with Visual Studio, you can install Visual Studio Community edition instead and use it to compile the sample programs. However, Rust is recommended for this course because the examples are written in Rust, and the language's explicit memory management makes it easier to understand what's happening at the assembly level.

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

Now compile your program using Cargo's release mode, which applies optimizations and produces a binary similar to what you'd encounter in real-world applications:

```powershell
cargo build --release
```

The `--release` flag is important because it enables optimizations that make the binary smaller and faster, but also more challenging to reverse engineer—exactly the kind of binaries you'll encounter in practice. The compilation process will take a few seconds as Rust compiles your code and links it with the necessary libraries.

The compiled binary will be located at: `target/release/hello_reversing.exe`. This is a fully functional Windows executable that you can run independently of the Rust toolchain.

### Test the Binary

Before analyzing the binary, verify that it works correctly by running it:

```powershell
.\target\release\hello_reversing.exe
```

The program will display the welcome message and prompt you to enter your name. Type a name and press Enter. The program should greet you and tell you how many characters are in your name. If you press Enter without typing anything, it should display "You didn't enter a name!" This simple program demonstrates conditional logic, string handling, and I/O operations—all of which will be interesting to observe at the assembly level later in the course.

## Inspecting Your Binary

Now that you have a working binary, it's time to take your first look at its internal structure using PE-bear, a tool specifically designed for examining Portable Executable (PE) files—the format used by Windows executables and DLLs.

Launch PE-bear from your Start menu or desktop shortcut. Once it opens, go to File → Open and navigate to your compiled `hello_reversing.exe` file (in the `target/release` directory of your Rust project). PE-bear will parse the file and display its structure in a hierarchical view.

Explore the various sections of the PE structure. The **DOS Header** is the first thing you'll see—this is a legacy structure dating back to MS-DOS compatibility. Look for the "MZ" signature (0x5A4D in hex), which identifies this as an executable file. The DOS header also contains a pointer to the actual PE header, which is where the modern executable format begins.

The **PE Header** contains critical information about the binary. You'll see the machine type listed as "AMD64" or "x64", indicating this is a 64-bit executable. The header also shows the number of sections in the binary, the timestamp when it was compiled, and various flags that control how Windows loads and executes the file.

The **Sections** view lists all the sections in your binary. You'll typically see `.text` (which contains the executable code), `.data` (initialized writable data), `.rdata` (read-only data like string constants), and possibly others like `.pdata` (exception handling information). Each section has attributes like size, virtual address, and permissions (readable, writable, executable). Understanding sections is crucial for reverse engineering because different types of data live in different sections.

The **Imports** section shows which external DLLs your binary depends on and which functions it imports from them. You'll see entries for `kernel32.dll` (core Windows API), `vcruntime140.dll` (Visual C++ runtime), and possibly others. For each DLL, you can expand the list to see individual imported functions like `GetStdHandle`, `WriteConsoleW`, `ReadConsoleW`, etc. These imports tell you what operating system functionality your program uses.

Finally, note the **Entry Point** RVA (Relative Virtual Address)—this is the address where Windows will begin executing your program. This isn't the `main` function you wrote; it's actually the runtime initialization code that sets up the environment before calling your `main` function. You'll learn more about this in [Lesson 3: PE Format Basics](lesson_03_pe_format_basics.md).

Take note of all these values and how they're organized. You'll encounter these same structures repeatedly throughout the course, and understanding the PE format is fundamental to Windows reverse engineering.

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

After completing the VM setup process, you should have a fully functional reverse engineering laboratory. Your Windows 10 or 11 virtual machine should be configured with at least 4 CPU cores (though 8 is better for performance when running multiple analysis tools simultaneously) and at least 8 GB of RAM (16 GB recommended to prevent slowdowns when debugging large applications).

All Windows updates should be installed, which you can verify by opening Settings → Windows Update and confirming that it shows "You're up to date" with no pending updates. This is important because outdated system libraries can cause compatibility issues with modern reverse engineering tools.

All reversing tools should be installed and verified to be working. You can confirm this by running the following verification commands in PowerShell:

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

Additionally, you should be able to launch Binary Ninja, PE-bear, Process Explorer, and Process Monitor from the Start menu. If any tool fails to launch or isn't found, revisit the installation steps for that specific tool.

Finally, you should have a snapshot named `clean-reversing-lab` (or similar) that captures the entire state of your VM at this point. You can verify this by checking your hypervisor's snapshot manager—the snapshot should appear in the list with the name and description you provided. This snapshot is your safety net for the entire course, allowing you to revert to a clean state whenever needed.

### Solution 2: Compiled Rust Program

Your compiled Rust program should be a fully functional Windows executable located at `target/release/my_first_binary.exe` (or whatever name you chose for your project). The binary should run without errors when executed from the command line or by double-clicking it in File Explorer.

When you run the program, it should display a welcome message, prompt for user input, and then process that input in some way. For example, if you followed the suggested exercise of creating a character counter, the interaction might look like this:

```
Welcome to Binary Reversing!
Enter your name:
Alice
Hello, Alice!
Your name has 5 characters.
```

The key learning point here is that you've created a real Windows executable from source code. You know exactly what this program does because you wrote it, which makes it an ideal target for learning reverse engineering—you can compare what you see in the disassembler with what you know the code should be doing. This "ground truth" is invaluable when you're first learning to read assembly.

If your program doesn't run correctly, check for compilation errors by reviewing the output of `cargo build --release`. Rust's compiler provides excellent error messages that usually point directly to the problem. Common issues include syntax errors, type mismatches, or incorrect use of the standard library.

### Solution 3: PE-bear Analysis

When you open your compiled binary in PE-bear, you should see a wealth of information about the PE file structure. Understanding what you're looking at is crucial for the rest of the course, so let's break down what each section means.

**DOS Header**: At the very beginning of the file, you'll see the DOS header with the magic number `MZ` (0x5A4D in hexadecimal). This is a signature that identifies the file as an executable. The DOS header also contains a pointer to the PE header, typically at offset 0x40 or nearby. This two-header structure exists for backward compatibility with MS-DOS, though modern Windows doesn't actually use the DOS portion.

**PE Header**: The PE header contains the real metadata about your executable. The Machine field should show `0x8664`, which indicates this is an x64 (64-bit) executable. The Number of Sections field typically shows 4-6 sections for a simple Rust program. The Characteristics field (usually `0x0022`) indicates that this is an executable file and that it's large address aware (meaning it can use the full 64-bit address space).

**Sections**: Rust binaries typically contain several standard sections. The `.text` section contains your executable code—this is where the compiled assembly instructions live. The `.data` section contains initialized writable data (global variables that have initial values). The `.rdata` section contains read-only data like string constants and lookup tables. You might also see `.reloc` (relocation information used when the binary is loaded at different addresses) and `.pdata` (exception handling metadata for x64 binaries).

Each section has attributes that control how Windows treats it. The `.text` section is marked as executable and readable but not writable (to prevent self-modifying code). The `.data` section is readable and writable but not executable. These permissions are enforced by the Windows memory manager and are an important security feature.

**Imports**: The imports section reveals which external functions your program depends on. For a simple Rust program, you'll typically see imports from `kernel32.dll`, which is the core Windows API library. Common functions include:

- `GetStdHandle`: Retrieves handles for standard input, output, and error streams
- `WriteFile` or `WriteConsoleW`: Writes data to the console
- `ReadFile` or `ReadConsoleW`: Reads data from the console
- `ExitProcess`: Terminates the program

You might also see imports from `ntdll.dll` (the Native API, which kernel32.dll itself calls) and `vcruntime140.dll` (the Visual C++ runtime library, which Rust uses for certain low-level operations).

Understanding these imports is valuable because they tell you what your program can do. A program that imports `CreateFileW` and `WriteFile` is likely doing file I/O. A program that imports `CreateThread` is using multithreading. As you progress through the course, you'll learn to use import analysis as a quick way to understand a binary's capabilities before diving into detailed disassembly.

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

