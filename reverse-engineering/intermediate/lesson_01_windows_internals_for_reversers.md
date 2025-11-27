# Lesson 1: Windows Internals for Reversers – Understanding Process Structure

## Overview

To reverse engineer Windows binaries effectively, you need to understand how Windows manages processes, memory, and threads. This lesson covers the internal structures that reversers encounter: the **PEB** (Process Environment Block), **TEB** (Thread Environment Block), and the **virtual memory layout**.

Understanding these structures helps you:
- Locate important data in memory
- Understand how the OS loads and manages your binary
- Recognize anti-debugging and anti-VM techniques
- Write more sophisticated hooks and patches

## What You'll Learn

By the end of this lesson, you will understand:

- **The PEB (Process Environment Block)** and its structure
- **The TEB (Thread Environment Block)** and its structure
- **Virtual memory layout** in Windows
- **Module lists** and how to enumerate loaded DLLs
- **How to access these structures** from a debugger or hook
- **Common anti-debugging techniques** that use these structures

## Prerequisites

Before starting this lesson, you should:

- Have completed the Beginner course
- Be comfortable with x86-64 assembly
- Understand pointers and memory layout

## The PEB (Process Environment Block)

The **PEB** is a structure maintained by Windows that contains information about a process. It includes:
- The image base (where the executable is loaded)
- The list of loaded modules (DLLs)
- The heap base
- Environment variables
- Command-line arguments

### Accessing the PEB

In x86-64, the PEB is accessed through the **GS register**:
- `GS:[0x60]` points to the PEB
- The PEB is at `GS:[0x60]` (on x64)

In assembly:
```asm
mov rax, gs:[0x60]    ; RAX now points to the PEB
```

### PEB Structure

The PEB structure (simplified) looks like:

```c
typedef struct _PEB {
    BYTE Reserved1[2];
    BYTE BeingDebugged;           // Offset 0x02
    BYTE Reserved2[1];
    PVOID Reserved3[2];
    PPEB_LDR_DATA Ldr;            // Offset 0x10 - Loader data
    PRTL_USER_PROCESS_PARAMETERS ProcessParameters;
    PVOID Reserved4[3];
    PVOID AtlThunkSListPtr;
    PVOID Reserved5;
    ULONG Reserved6;
    PVOID Reserved7;
    ULONG Reserved8;
    ULONG AtlThunkSListPtr32;
    PVOID Reserved9[45];
    BYTE Reserved10[96];
    PPS_POST_PROCESS_INIT_ROUTINE PostProcessInitRoutine;
    PVOID Reserved11[128];
    PVOID Reserved12[1];
    ULONG SessionId;
} PEB, *PPEB;
```

### Important PEB Fields

- **BeingDebugged** (offset 0x02): Set to 1 if the process is being debugged
- **Ldr** (offset 0x10): Points to the loader data, which contains the module list
- **ImageBaseAddress** (offset 0x10): The base address where the executable is loaded

## The TEB (Thread Environment Block)

The **TEB** is a structure that contains thread-specific information. Each thread has its own TEB.

### Accessing the TEB

In x86-64, the TEB is accessed through the **GS register**:
- `GS:[0x30]` points to the TEB
- The TEB is at `GS:[0x30]` (on x64)

In assembly:
```asm
mov rax, gs:[0x30]    ; RAX now points to the TEB
```

### TEB Structure

The TEB structure (simplified) looks like:

```c
typedef struct _TEB {
    PVOID Reserved1[12];
    PPEB ProcessEnvironmentBlock;  // Offset 0x60 - Points to PEB
    PVOID Reserved2[399];
    BYTE Reserved3[1952];
    PVOID TlsSlots[64];            // Thread Local Storage
    BYTE Reserved4[8];
    PVOID Reserved5[10];
    ULONG LastErrorValue;
    // ... more fields
} TEB, *PTEB;
```

### Important TEB Fields

- **ProcessEnvironmentBlock** (offset 0x60): Points to the PEB
- **TlsSlots**: Thread Local Storage slots
- **LastErrorValue**: The last error code (from GetLastError)

## Virtual Memory Layout

Windows processes have a virtual memory layout that looks like:

```
0x0000000000000000 - 0x0000000000001000  : NULL page (inaccessible)
0x0000000000001000 - 0x0000000140000000  : User-mode memory
0x0000000140000000 - 0x0000000180000000  : Typically where executables are loaded
0x0000000180000000 - 0x0000000200000000  : Typically where DLLs are loaded
0x0000000200000000 - 0x7FFFFFFFFFFFFFFF  : More user-mode memory
0x8000000000000000 - 0xFFFFFFFFFFFFFFFF  : Kernel-mode memory (inaccessible from user mode)
```

### Image Base

The **image base** is where the executable is loaded. By default:
- 32-bit executables: 0x00400000
- 64-bit executables: 0x140000000

However, with ASLR (Address Space Layout Randomization), the image base is randomized.

## Module Lists

The PEB contains a linked list of loaded modules. You can traverse this list to find all loaded DLLs.

### Traversing the Module List

```c
// Pseudo-code to traverse the module list
PEB* peb = (PEB*)__readgsqword(0x60);
PEB_LDR_DATA* ldr = peb->Ldr;
LIST_ENTRY* head = &ldr->InLoadOrderModuleList;
LIST_ENTRY* current = head->Flink;

while (current != head) {
    LDR_DATA_TABLE_ENTRY* entry = (LDR_DATA_TABLE_ENTRY*)current;
    // entry->DllBase points to the loaded DLL
    // entry->BaseDllName is the DLL name
    current = current->Flink;
}
```

## Anti-Debugging Techniques Using PEB

Many anti-debugging techniques check the PEB:

### Technique 1: Check BeingDebugged

```c
PEB* peb = (PEB*)__readgsqword(0x60);
if (peb->BeingDebugged) {
    // Debugger detected!
    exit(1);
}
```

### Technique 2: Check for Debugger Breakpoints

Some anti-debugging code checks for common debugger breakpoints in the module list.

### Technique 3: Check Heap Flags

The PEB contains heap information. Debuggers sometimes modify heap flags, which can be detected.

## Exercises

### Exercise 1: Access the PEB from a Debugger

**Objective**: Learn to access and inspect the PEB.

**Steps**:
1. Open a binary in x64dbg
2. Set a breakpoint at the entry point
3. In the registers panel, look at the GS register
4. Calculate the PEB address: `GS + 0x60`
5. In the memory view, navigate to the PEB address
6. Inspect the PEB structure
7. Find the BeingDebugged flag
8. Find the Ldr pointer

**Verification**: You should be able to locate and inspect the PEB.

### Exercise 2: Detect a Debugger Using PEB

**Objective**: Write code that detects a debugger.

**Steps**:
1. Write a Rust program that:
   - Accesses the PEB
   - Checks the BeingDebugged flag
   - Prints a message if a debugger is detected
2. Compile and run it normally (should not detect debugger)
3. Run it in x64dbg (should detect debugger)

**Verification**: The program should detect the debugger when run in x64dbg.

### Exercise 3: Enumerate Loaded Modules

**Objective**: Learn to traverse the module list.

**Steps**:
1. Write a Rust program that:
   - Accesses the PEB
   - Traverses the module list
   - Prints the name and base address of each loaded DLL
2. Compile and run it
3. Verify the output matches what you see in Process Explorer

**Verification**: Your program should list all loaded DLLs.

## Solutions

### Solution 1: Access the PEB from a Debugger

When you set a breakpoint and inspect the PEB:
1. The GS register contains the base address of the TEB
2. At offset 0x60 from the TEB is a pointer to the PEB
3. The PEB contains the BeingDebugged flag at offset 0x02
4. The Ldr pointer is at offset 0x10

### Solution 2: Detect a Debugger Using PEB

A simple debugger detection program:

```rust
unsafe {
    let peb = *(0x60 as *const *const u8);
    let being_debugged = *(peb.add(0x02) as *const u8);
    if being_debugged != 0 {
        println!("Debugger detected!");
    }
}
```

### Solution 3: Enumerate Loaded Modules

A program to enumerate modules:

```rust
unsafe {
    let peb = *(0x60 as *const *const u8);
    let ldr = *(peb.add(0x10) as *const *const u8);
    // Traverse the module list...
}
```

## Summary

You now understand Windows process internals. You can:

- Access and inspect the PEB and TEB
- Understand virtual memory layout
- Traverse the module list
- Recognize anti-debugging techniques
- Implement debugger detection

In the next lesson, you'll learn about exception handling and TLS callbacks.
