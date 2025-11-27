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

The **Process Environment Block (PEB)** is one of the most important data structures in Windows reverse engineering. It's a structure maintained by Windows in user-mode memory that contains critical information about a running process. Understanding the PEB is essential because malware and protected software frequently access it to gather information about the process environment, detect debuggers, or locate loaded modules.

The PEB contains a wealth of information that's useful for both legitimate programs and malware. It includes the image base address (where the executable is loaded in memory), which is essential for calculating absolute addresses from relative addresses. It contains the list of loaded modules (DLLs), allowing programs to enumerate what libraries are loaded without calling Windows APIs. The heap base address is stored here, providing access to the process's heap. Environment variables and command-line arguments are also accessible through the PEB, making it a one-stop shop for process information.

### Accessing the PEB

On x86-64 Windows, the PEB is accessed through the **GS segment register**, which is a special CPU register that points to thread-local storage. The GS register is used differently on x64 than on x86 (where the FS register is used instead). Specifically, the address at `GS:[0x60]` contains a pointer to the PEB structure.

This access method is consistent across all x64 Windows processes, making it a reliable way to locate the PEB. In assembly, accessing the PEB looks like this:

```asm
mov rax, gs:[0x60]    ; RAX now points to the PEB
```

After this instruction, RAX contains the address of the PEB structure, and you can access any field within it by adding the appropriate offset. For example, to check the `BeingDebugged` flag at offset 0x02, you would use `mov al, [rax+0x02]`.

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

The PEB structure contains many fields, but several are particularly important for reverse engineering and are frequently accessed by both legitimate software and malware.

**BeingDebugged** (offset 0x02) is a single byte that's set to 1 if the process is being debugged, and 0 otherwise. This is the field that `IsDebuggerPresent()` checks. Malware frequently reads this field directly to detect debuggers without calling the API, making it harder to hook. You can bypass this by manually setting the byte to 0 in your debugger.

**Ldr** (offset 0x18 on x64) is a pointer to the `PEB_LDR_DATA` structure, which contains the loader data. This structure includes linked lists of all loaded modules (DLLs), organized in three different orders: load order, memory order, and initialization order. By traversing these lists, you can enumerate all DLLs loaded in the process without calling any Windows APIs—a technique commonly used by malware to locate functions for dynamic resolution.

**ImageBaseAddress** (offset 0x10) contains the base address where the main executable is loaded in memory. This is useful for calculating absolute addresses from relative virtual addresses (RVAs) found in the PE file. With ASLR (Address Space Layout Randomization) enabled, this value changes each time the process runs, but you can always find the current base address by reading this field.

## The TEB (Thread Environment Block)

The **Thread Environment Block (TEB)** is the thread-level equivalent of the PEB. While the PEB contains process-wide information, the TEB contains information specific to a single thread. Every thread in a process has its own TEB, making it essential for understanding multi-threaded programs and thread-local storage.

The TEB is particularly important for reverse engineering because it provides access to thread-local data, exception handlers, and the thread's stack boundaries. Malware often uses the TEB to access thread-specific information or to manipulate exception handling.

### Accessing the TEB

Like the PEB, the TEB is accessed through the GS segment register on x64 Windows. However, it's at a different offset: `GS:[0x30]` contains a pointer to the current thread's TEB. This means each thread, when it accesses `GS:[0x30]`, gets a pointer to its own TEB, not to other threads' TEBs.

In assembly, accessing the TEB looks like this:

```asm
mov rax, gs:[0x30]    ; RAX now points to the current thread's TEB
```

Interestingly, the TEB also contains a pointer back to the PEB at offset 0x60, providing an alternative way to access the PEB: `mov rax, gs:[0x30]` followed by `mov rax, [rax+0x60]` will give you the PEB address.

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

The TEB contains numerous fields, but several are particularly relevant for reverse engineering and understanding program behavior.

**ProcessEnvironmentBlock** (offset 0x60) is a pointer back to the PEB. This provides an alternative way to access the PEB from the TEB, which can be useful in certain contexts. Some code accesses the PEB through the TEB rather than directly through `GS:[0x60]`, so recognizing this pattern is important.

**TlsSlots** (offset varies) is an array of 64 pointers used for Thread Local Storage (TLS). TLS allows each thread to have its own copy of certain variables. When you see code accessing `TlsSlots`, it's typically reading or writing thread-specific data. Malware sometimes uses TLS to store per-thread state or to hide data from analysis.

**LastErrorValue** (offset varies) stores the last error code set by Windows APIs. This is the value returned by `GetLastError()`. Understanding that this is stored in the TEB explains why `GetLastError()` is thread-safe—each thread has its own error value. When debugging, you can inspect this field to see what error occurred without calling `GetLastError()`.

**StackBase** and **StackLimit** (offsets vary) define the boundaries of the thread's stack. These are useful for understanding stack overflows, validating stack pointers, or implementing custom stack walking. Some anti-debugging techniques check these values to detect stack manipulation.

## Virtual Memory Layout

Understanding the virtual memory layout of Windows processes is crucial for reverse engineering because it helps you understand where different components are loaded and why certain addresses look the way they do. Windows uses a 64-bit address space on x64 systems, but not all of it is usable—the address space is divided into user-mode and kernel-mode regions.

The virtual memory layout on x64 Windows looks like this:

```
0x0000000000000000 - 0x0000000000001000  : NULL page (inaccessible)
0x0000000000001000 - 0x0000000140000000  : User-mode memory
0x0000000140000000 - 0x0000000180000000  : Typically where executables are loaded
0x0000000180000000 - 0x0000000200000000  : Typically where DLLs are loaded
0x0000000200000000 - 0x7FFFFFFFFFFFFFFF  : More user-mode memory
0x8000000000000000 - 0xFFFFFFFFFFFFFFFF  : Kernel-mode memory (inaccessible from user mode)
```

The NULL page (the first 4KB) is always inaccessible, which is why dereferencing a NULL pointer causes an access violation. This is a deliberate design choice to catch NULL pointer bugs.

User-mode memory occupies the lower half of the address space (addresses starting with 0x0000...). This is where your program's code, data, heap, and stack live. Any attempt to access kernel-mode memory (addresses starting with 0x8000... or higher) from user mode will cause an access violation.

### Image Base

The **image base** is the address where the executable is loaded into memory. This is a critical concept because all addresses in the PE file are relative to the image base. When you see an RVA (Relative Virtual Address) in the PE file, you add it to the image base to get the actual memory address.

By default, Windows uses predictable image base addresses. For 32-bit executables, the default is 0x00400000, which is why you often see addresses like 0x00401000 in x86 programs. For 64-bit executables, the default is 0x140000000, which is why x64 programs typically have addresses like 0x140001000.

However, with ASLR (Address Space Layout Randomization) enabled—which is the default for modern Windows binaries—the image base is randomized each time the process starts. This security feature makes it harder for exploits to predict where code will be located. When reversing ASLR-enabled binaries, you need to account for the fact that addresses will be different each time you run the program. You can find the actual image base by reading the PEB's `ImageBaseAddress` field or by checking the base address in your debugger.

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
