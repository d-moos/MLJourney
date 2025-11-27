# Lesson 3: Simple Packers and Manual Unpacking – Reversing Packed Binaries

## Overview

A **packer** is a tool that compresses and encrypts a binary to make it harder to reverse engineer. When a packed binary runs, it unpacks itself in memory and executes the original code.

Understanding packers is important because:
- Many malware samples are packed
- Game protections use packers
- Understanding unpacking helps you analyze protected binaries
- Manual unpacking is a fundamental skill

## What You'll Learn

By the end of this lesson, you will understand:

- **How packers work** (stub + encrypted payload)
- **How to identify packed binaries**
- **How to find the Original Entry Point (OEP)**
- **How to dump unpacked code from memory**
- **How to rebuild the IAT** (Import Address Table)
- **How to use tools like Scylla** for IAT reconstruction

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-2 of this course
- Understand PE file structure
- Be comfortable with x64dbg

## How Packers Work

A packer works in three stages:

1. **Stub**: The packer's code that unpacks the original binary
2. **Encrypted Payload**: The original binary, encrypted
3. **Unpacking**: The stub decrypts the payload and executes it

### Packer Structure

```
Packed Binary:
  [Packer Stub]
  [Encrypted Original Binary]
  [Packer Data]
```

When the packed binary runs:
1. The packer stub executes
2. The stub decrypts the original binary in memory
3. The stub jumps to the Original Entry Point (OEP)
4. The original binary executes

## Identifying Packed Binaries

Before you can unpack a binary, you need to recognize that it's packed in the first place. Packed binaries have distinctive characteristics that set them apart from normal executables. Learning to spot these signs quickly is an essential skill for malware analysis and reverse engineering protected software.

### Signs of Packing

**Few imports** is one of the most obvious indicators of packing. A normal Windows executable imports dozens or even hundreds of functions from various DLLs—functions for file I/O, memory management, GUI operations, networking, and more. A packed binary, however, typically imports only a handful of functions, often just the bare minimum needed for the unpacking stub to work. You might see only `LoadLibraryA`, `GetProcAddress`, and `VirtualAlloc`—just enough to allocate memory, decrypt the payload, and dynamically resolve the real imports. If you open a binary in PE-bear or Binary Ninja and see fewer than 10 imports, that's a strong indicator of packing.

**Suspicious entry point** location is another telltale sign. In a normal binary, the entry point is typically in the `.text` section, which contains the program's code. In a packed binary, the entry point might be in an unusually named section like `.packed`, `.stub`, `.upx0`, or even a section with a random or blank name. This happens because the packer adds its own code section to run the unpacking routine before jumping to the original code.

**High entropy** in certain sections indicates encryption or compression. Entropy is a measure of randomness—uncompressed, unencrypted code and data have relatively low entropy because they contain patterns and structure. Encrypted or compressed data, however, looks essentially random and has high entropy approaching the theoretical maximum. Tools like Binary Ninja can calculate entropy for each section. If you see a large section with entropy close to 8.0 (the maximum for byte-level entropy), that section likely contains encrypted or compressed data.

**Small code section** combined with a **large data section** is a classic packing pattern. The `.text` section might be only a few kilobytes—just enough for the unpacking stub—while there's a huge `.data` or custom section containing the encrypted original binary. This is the opposite of what you'd expect in a normal program, where the code section is typically larger than the data sections.

### Using Binary Ninja to Detect Packing

Binary Ninja provides excellent tools for detecting packed binaries. When you load a suspicious binary, start by examining the sections view. Open the binary in Binary Ninja and navigate to the sections panel (usually visible in the sidebar or accessible through View → Sections).

Look at the list of sections and note their names, sizes, and attributes. Normal Windows binaries have predictable section names like `.text`, `.data`, `.rdata`, `.pdata`, and `.reloc`. If you see unusual section names or sections with generic names like `UPX0`, `UPX1`, or completely blank names, that's suspicious.

Next, check the entropy of each section. Binary Ninja can display entropy information in the section properties. Look for sections with high entropy (above 7.0 or so). High entropy in a large section strongly suggests that section contains encrypted or compressed data. Compare this with the `.text` section of a normal binary, which typically has entropy around 5.0-6.5 due to the patterns in x86-64 machine code.

Finally, examine the imports. Go to the imports view and count how many functions are imported. A normal application might import 50-200 functions or more. A packed binary might import only 5-10 functions, and they're usually memory-related functions like `VirtualAlloc`, `VirtualProtect`, and dynamic loading functions like `LoadLibraryA` and `GetProcAddress`. This minimal import table is a dead giveaway—the packer only imports what it needs to unpack and resolve the real imports at runtime.

## Finding the Original Entry Point (OEP)

The **Original Entry Point (OEP)** is the address where the original, unpacked binary begins executing. Finding the OEP is the critical step in manual unpacking because it's the point at which the packer has finished its work and the real program is about to start. Once you've found the OEP, you can dump the unpacked code from memory and rebuild a working executable.

### Method 1: Breakpoint on Entry Point and Step Through

This is the most reliable method for finding the OEP, though it requires patience and careful observation. The technique involves stepping through the packer's code until you see it transfer control to the original program.

Start by opening the packed binary in x64dbg. The debugger will automatically pause at the entry point—the first instruction of the packer stub. Set a breakpoint here if one isn't already set (though x64dbg usually breaks automatically on load).

Now comes the tedious but educational part: step through the packer code. Use F8 (Step Over) to execute instructions one at a time. As you step through, watch for characteristic packer behavior. You'll typically see:

- Memory allocation calls like `VirtualAlloc` or `VirtualProtect` (the packer is allocating space for the unpacked code)
- Loops that read from one memory location and write to another (the decryption/decompression loop)
- Calls to `GetProcAddress` or similar functions (the packer is resolving imports for the unpacked binary)

Continue stepping until you see a large `jmp` or `call` instruction that jumps to an address far from the current code. This is often the transition from packer to original code. The destination address is likely the OEP. You can verify this by looking at the destination—if it's in a different section or at a much higher/lower address than the packer code, and if the code there looks like a normal function prologue (like `push rbp; mov rbp, rsp` or `sub rsp, XX`), you've probably found the OEP.

Some packers use a `push <address>; ret` sequence instead of a direct `jmp`, which has the same effect but is slightly more obfuscated. Watch for any instruction that dramatically changes RIP to a new region of memory.

### Method 2: Look for Suspicious Jumps in Static Analysis

If you want to get a hint about where the OEP might be before running the binary, you can use static analysis to look for suspicious jumps. This method is faster but less reliable than dynamic analysis.

Open the packed binary in Binary Ninja and navigate to the entry point. Read through the disassembly looking for control flow instructions that jump to unusual addresses. A normal function has relatively local jumps—conditional branches that jump a few instructions forward or backward, and calls to nearby functions. A packer, however, will eventually have a jump that goes to a completely different address range.

Look for `jmp` instructions with large immediate offsets, or `jmp` instructions that use a register containing a computed address. For example, you might see something like `mov rax, <large address>; jmp rax`. The address in that instruction is a candidate for the OEP.

This method is less reliable because modern packers often obfuscate the control flow, using indirect jumps through registers or memory locations that are only computed at runtime. Static analysis can give you hints, but you'll usually need to verify your findings with dynamic analysis.

### Method 3: Use Entropy Analysis and Memory Breakpoints

This is a more advanced technique that leverages the fact that encrypted data has high entropy, but executable code has lower entropy. The idea is to watch for when high-entropy data is transformed into low-entropy code.

The packer stub itself has low entropy because it's normal executable code. The encrypted payload has high entropy because it's encrypted. When the packer decrypts the payload in memory, that memory region transitions from high entropy to low entropy. If you can detect this transition, you know where the unpacked code is located.

In practice, this is done by setting memory breakpoints on the region where you suspect the code is being unpacked. In x64dbg, you can set a memory breakpoint on execution (break when code executes from a specific memory region). Set this breakpoint on the memory region that was allocated by `VirtualAlloc`, and when the breakpoint triggers, you're likely at or very near the OEP.

Another approach is to set a breakpoint on `VirtualProtect`, which packers often call to change memory permissions from writable (during unpacking) to executable (before jumping to the OEP). When `VirtualProtect` is called to make a region executable, that region likely contains the unpacked code, and the OEP is probably nearby.

## Dumping Unpacked Code

Once you've found the OEP, the next step is to dump the unpacked code from memory to a file. At this point, the original binary is sitting in memory in its unpacked form, but it's not yet saved to disk. Dumping creates a file that you can analyze with static analysis tools.

Set a breakpoint at the OEP address you identified. Run the program (F9) until it hits this breakpoint. At this point, the unpacking is complete, and the original code is fully decrypted and ready to execute. This is the perfect moment to dump the memory.

In x64dbg, you need to dump the entire module, not just the code at the current address. Right-click in the CPU view and select "Follow in Memory Map" to see all the memory regions allocated to the process. Find the region that contains the unpacked code—this is usually the region that starts at the module's base address (you can see this in the Memory Map tab).

Right-click on the memory region containing the unpacked code and select "Dump Memory to File". x64dbg will prompt you for a filename. Save it with a `.exe` or `.dll` extension as appropriate. This creates a raw dump of the memory region, which contains the unpacked PE file.

However, this dumped file might not run correctly yet. The problem is that the Import Address Table (IAT) might not be properly reconstructed in the dump. Many packers dynamically resolve imports at runtime, so the IAT in memory is different from the IAT in the original packed file. This is where IAT reconstruction comes in.

## Rebuilding the IAT

When you dump unpacked code from memory, the Import Address Table (IAT) often needs to be rebuilt. The IAT is a table of pointers to imported functions—when your program calls an imported function like `CreateFileW`, it's actually calling through the IAT. Packers often destroy or encrypt the original IAT and rebuild it at runtime using `GetProcAddress`. Your dumped file needs a proper IAT to run correctly.

### Using Scylla

Scylla is a powerful tool specifically designed for IAT reconstruction. It's available as a standalone application or as a plugin for x64dbg, making it perfect for unpacking workflows. Scylla can automatically scan memory to find the IAT, identify all imported functions, and rebuild the import directory in your dumped file.

To use Scylla, first ensure your program is paused at the OEP in x64dbg—this is important because the IAT needs to be fully resolved at this point. Open Scylla (if it's installed as a plugin, you can access it from the Plugins menu in x64dbg; otherwise, run it as a standalone application and attach it to the x64dbg process).

In the Scylla interface, you'll see fields for the OEP and IAT address. Click "IAT Autosearch" to have Scylla automatically scan the process memory looking for the Import Address Table. Scylla uses heuristics to identify the IAT—it looks for regions of memory containing pointers to known DLL functions. When it finds the IAT, it will display the address and size.

Next, click "Get Imports". This tells Scylla to analyze the IAT and identify all the imported functions. Scylla will read the pointers in the IAT, determine which DLL and function each pointer refers to, and build a list of imports. You'll see this list populate in the Scylla interface, showing DLL names and function names.

Review the import list to make sure it looks reasonable. You should see familiar Windows DLLs like `kernel32.dll`, `ntdll.dll`, `user32.dll`, etc., along with their imported functions. If you see a lot of invalid or unknown imports, the IAT autosearch might have found the wrong region, and you may need to manually specify the IAT address.

Once you're satisfied with the import list, click "Dump" and select your dumped file (the one you created earlier with "Dump Memory to File"). Scylla will modify the PE file to include a proper import directory that matches the imports it found in memory. Save the fixed binary with a new name (like `unpacked_fixed.exe`).

The resulting file should now be a fully functional unpacked binary that you can analyze with static analysis tools like Binary Ninja, and it should run independently without needing the packer.

## Exercises

### Exercise 1: Identify a Packed Binary

**Objective**: Learn to identify packed binaries.

**Steps**:
1. Download a simple packed binary (or create one using UPX)
2. Open it in Binary Ninja
3. Analyze the sections
4. Check the entropy
5. Look at the imports
6. Document signs of packing

**Verification**: You should be able to identify that the binary is packed.

### Exercise 2: Find the OEP

**Objective**: Learn to find the Original Entry Point.

**Steps**:
1. Open a packed binary in x64dbg
2. Set a breakpoint at the entry point
3. Run the program
4. Step through the packer code
5. Look for a `jmp` or `call` to a new address
6. Document the OEP address

**Verification**: You should find the OEP address.

### Exercise 3: Dump and Rebuild

**Objective**: Learn to dump and rebuild a packed binary.

**Steps**:
1. Open a packed binary in x64dbg
2. Find the OEP
3. Set a breakpoint at the OEP
4. Run the program
5. Dump the unpacked code to a file
6. Use Scylla to rebuild the IAT
7. Verify the unpacked binary works

**Verification**: The unpacked binary should run correctly.

## Solutions

### Solution 1: Identify a Packed Binary

When you analyze a packed binary using the techniques described above, you should observe several characteristic indicators that distinguish it from a normal executable. Let's walk through what you'd see with a typical UPX-packed binary as an example.

**Import Analysis**: Open the binary in PE-bear or Binary Ninja and examine the imports section. A packed binary will have dramatically fewer imports than a normal program. You might see only 3-5 imported functions, typically including `LoadLibraryA` and `GetProcAddress` from `kernel32.dll`. These two functions are essential for the packer because they allow it to dynamically load DLLs and resolve function addresses at runtime, rebuilding the original program's import table. You might also see `VirtualAlloc`, `VirtualProtect`, or `VirtualFree`—memory management functions the packer needs to allocate space for the unpacked code and change memory permissions.

Compare this to a normal binary, which might import 50-200 functions covering file I/O (`CreateFileW`, `ReadFile`, `WriteFile`), console operations (`WriteConsoleW`), process management (`CreateProcess`), and many others. The stark difference in import count is one of the most reliable indicators of packing.

**Entropy Analysis**: Using Binary Ninja's entropy analysis features, examine each section of the binary. The `.text` section (or whatever the packer calls its code section) will have relatively low entropy—around 5.0-6.5—because it contains normal x86-64 machine code with predictable patterns. However, you'll find another section (often called `.upx1`, `.packed`, or something similar) with very high entropy—7.5 or higher, approaching the theoretical maximum of 8.0. This high entropy indicates that the section contains encrypted or compressed data that appears essentially random. This is the encrypted original binary.

**Section Analysis**: Look at the section sizes and characteristics. You'll typically see a small code section (perhaps 5-20 KB) containing just the unpacking stub, and a much larger data section (potentially hundreds of KB or even MB) containing the encrypted payload. This is the inverse of what you'd expect in a normal binary, where the code section is usually larger than the data sections. Additionally, the section names might be unusual—UPX uses `.upx0` and `.upx1`, other packers might use generic names or even blank names.

**Entry Point**: Check where the entry point is located. In PE-bear, you can see the entry point RVA in the Optional Header. Cross-reference this with the sections to see which section contains the entry point. In a packed binary, the entry point is often in an unusual section rather than the standard `.text` section. This is because the packer's stub code runs first, before the original program.

### Solution 2: Find the OEP

Finding the Original Entry Point requires patience and careful observation as you step through the packer's code. Here's what you should see during a typical unpacking session with a simple packer like UPX.

**Initial Execution**: When you first load the packed binary in x64dbg, you'll be at the entry point of the packer stub. The code here might look unusual compared to normal program initialization—instead of the typical function prologue and setup code, you might see immediate jumps, unusual register manipulations, or anti-debugging checks.

**Decryption Loops**: As you step through (using F8 to step over calls), you'll eventually encounter loops that perform the actual decryption or decompression. These loops typically read from one memory location (the encrypted data), perform some transformation (XOR, bit rotation, decompression algorithm), and write to another location (the destination for the unpacked code). You might see patterns like:

```
loop_start:
    mov al, [rsi]      ; Read encrypted byte
    xor al, bl         ; Decrypt it
    mov [rdi], al      ; Write decrypted byte
    inc rsi
    inc rdi
    dec rcx
    jnz loop_start     ; Continue until done
```

These loops can execute thousands or millions of times, so don't step through them instruction by instruction—use breakpoints or step over the entire loop.

**Memory Allocation**: Before or during the decryption process, you'll see calls to `VirtualAlloc` or similar functions. The packer is allocating memory to hold the unpacked code. Note the address returned by `VirtualAlloc`—this is likely where the unpacked code will reside, and the OEP will be somewhere in this region.

**Import Resolution**: After decryption, many packers rebuild the Import Address Table by calling `LoadLibraryA` to load DLLs and `GetProcAddress` to resolve function addresses. You might see loops that iterate through a list of DLL names and function names, calling these functions repeatedly. This is the packer reconstructing the original program's imports.

**The Final Jump**: Eventually, you'll reach the moment of transition from packer to original code. This typically appears as a `jmp` instruction to an address far from the current code, or a `push <address>; ret` sequence that has the same effect. For example:

```
mov rax, 0x140001234
jmp rax
```

or

```
push 0x140001234
ret
```

The destination address (0x140001234 in these examples) is the OEP. When you see this jump, note the address. You can verify it's the OEP by looking at the code at that address—it should look like a normal function prologue, such as `push rbp; mov rbp, rsp; sub rsp, 0x20`, which is typical for the start of a function.

### Solution 3: Dump and Rebuild

After successfully dumping and rebuilding the IAT, you should have a fully functional unpacked binary that can be analyzed and executed independently. Here's what success looks like and how to verify it.

**Correct Imports**: Open your rebuilt binary in PE-bear or Binary Ninja and examine the imports section. Unlike the original packed binary (which had only 3-5 imports), the unpacked binary should now have a complete import table with all the functions the original program uses. You might see dozens or hundreds of imports from various DLLs like `kernel32.dll`, `user32.dll`, `ntdll.dll`, `msvcrt.dll`, etc.

The import table should list specific functions that make sense for the program's functionality. For example, if it's a GUI application, you should see imports like `CreateWindowExW`, `ShowWindow`, `GetMessageW`, etc. If it's a console application, you should see `GetStdHandle`, `WriteConsoleW`, `ReadConsoleW`, etc. The presence of these detailed imports indicates that Scylla successfully reconstructed the IAT.

**Binary Execution**: The ultimate test is whether the unpacked binary runs correctly. Try executing it (in your VM, of course—never run untrusted binaries on your host machine). The program should behave identically to the original packed version. If it's a simple program that prints "Hello, World!", it should still print that message. If it's a more complex application with a GUI, the GUI should appear and function normally.

If the binary crashes immediately or displays an error about missing imports, the IAT reconstruction may have failed. Common issues include:
- Scylla found the wrong IAT address (try manually specifying the IAT location)
- The dump was taken at the wrong point (make sure you're at the OEP, not in the middle of initialization)
- The binary uses delay-loaded imports or other advanced import mechanisms that Scylla doesn't handle automatically

**Static Analysis**: Open the unpacked binary in Binary Ninja and verify that you can now perform meaningful static analysis. Unlike the packed version (which showed only the packer stub code), the unpacked version should show the actual program logic. You should be able to:
- Navigate through functions and see real program code, not just decryption loops
- Find strings that the program uses (these were encrypted in the packed version)
- Trace control flow and understand what the program does
- Identify interesting functions and analyze their behavior

The ability to perform static analysis is the whole point of unpacking—you've transformed an opaque, encrypted binary into a readable program that you can analyze with all your normal tools. This is a crucial skill for malware analysis, where packers are used to hide malicious functionality, and for analyzing protected commercial software.

## Summary

You now understand packers and unpacking. You can:

- Identify packed binaries
- Find the Original Entry Point
- Dump unpacked code from memory
- Rebuild the IAT using Scylla
- Analyze unpacked binaries

In the next lesson, you'll learn about control flow obfuscation.
