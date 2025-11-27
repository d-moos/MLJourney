# Lesson 5: Manually Mapped DLLs – Analyzing Injected Code

## Overview

**Manually mapped DLLs** are DLLs that are loaded without using the standard Windows loader. Instead, the code manually:
1. Allocates memory
2. Copies the DLL into memory
3. Resolves imports
4. Calls the entry point

This technique is used by:
- Malware to hide DLL loading
- Game cheats to inject code
- Rootkits to hide from detection

## What You'll Learn

By the end of this lesson, you will understand:

- **How manually mapped DLLs work**
- **How to identify manually mapped DLLs**
- **How to analyze manually mapped DLLs**
- **How to reconstruct the DLL**
- **How to extract the DLL from memory**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-4 of this course
- Understand DLL structure
- Understand memory layout

## How Manually Mapped DLLs Work

Manually mapping a DLL is the process of loading a DLL into a process without using the Windows loader (`LoadLibrary`). This technique is commonly used by game cheats, malware, and rootkits because it bypasses many detection mechanisms that monitor `LoadLibrary` calls or enumerate loaded modules. Understanding manual mapping is essential for detecting and analyzing sophisticated malware and anti-cheat bypasses.

### Steps

The manual mapping process involves replicating what the Windows loader does, but doing it manually in your own code. This gives you complete control over how and where the DLL is loaded.

**Allocate memory** is the first step. You need to allocate a region of memory large enough to hold the entire DLL. Use `VirtualAlloc` or `VirtualAllocEx` (if injecting into another process) to allocate memory with read, write, and execute permissions. The size should be at least as large as the DLL's `SizeOfImage` field from the PE optional header. Unlike normal DLL loading, you can allocate this memory anywhere in the process's address space—it doesn't have to be at the DLL's preferred base address.

**Copy DLL** involves copying the DLL's sections into the allocated memory. You can't just copy the entire DLL file as-is because PE files on disk have a different layout than PE files in memory. Instead, you need to copy each section individually to its correct virtual address offset. Read the PE headers to find each section's file offset and virtual address, then copy the section data from the file to the corresponding location in your allocated memory. Don't forget to copy the PE headers themselves to the beginning of the allocated memory.

**Resolve imports** is one of the most critical steps. The DLL's import table contains references to functions in other DLLs, but these references are just names or ordinals—they're not actual addresses. You need to manually resolve each import by calling `GetModuleHandle` to get the handle of the imported DLL, then `GetProcAddress` to get the address of each imported function. Write these addresses into the Import Address Table (IAT) in your mapped DLL. This is exactly what the Windows loader does, but you're doing it manually.

**Fix relocations** is necessary if the DLL wasn't loaded at its preferred base address (which is almost always the case). The DLL contains a relocation table that lists all the addresses in the code and data that need to be adjusted based on where the DLL is actually loaded. For each relocation entry, calculate the delta (difference between actual base and preferred base), then add this delta to the value at the relocation address. This fixes all the hardcoded addresses in the DLL to point to the correct locations in memory.

**Call entry point** is the final step. Once everything is set up, call the DLL's `DllMain` function with the `DLL_PROCESS_ATTACH` reason. The entry point address is specified in the PE optional header. Cast this address to a function pointer with the `DllMain` signature and call it. If `DllMain` returns TRUE, the DLL has successfully initialized and is ready to use.

### Example Code

```c
// Allocate memory
PVOID dll_base = VirtualAlloc(NULL, dll_size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

// Copy DLL
memcpy(dll_base, dll_data, dll_size);

// Resolve imports
// ... (manually resolve each imported function)

// Fix relocations
// ... (apply relocations)

// Call entry point
typedef BOOL (*DllMainFunc)(HINSTANCE, DWORD, LPVOID);
DllMainFunc dll_main = (DllMainFunc)((PBYTE)dll_base + entry_point_offset);
dll_main((HINSTANCE)dll_base, DLL_PROCESS_ATTACH, NULL);
```

## Identifying Manually Mapped DLLs

Detecting manually mapped DLLs is challenging because they're specifically designed to evade normal DLL enumeration techniques. However, they leave characteristic traces that you can identify with the right tools and techniques.

### Signs of Manual Mapping

Several indicators can reveal the presence of manually mapped DLLs in a process.

**Unusual memory allocation** is often the first clue. Manually mapped DLLs require large memory allocations (typically hundreds of KB to several MB) with execute permissions. If you see `VirtualAlloc` or `VirtualAllocEx` calls allocating large regions with `PAGE_EXECUTE_READWRITE` or `PAGE_EXECUTE_READ` permissions, it might be for manual mapping. Normal programs rarely allocate large executable regions—they load DLLs through the Windows loader instead.

**Manual import resolution** is a telltale sign. The manual mapping code needs to resolve imports, so you'll see many calls to `GetModuleHandle` and `GetProcAddress` in sequence. If you see code that's calling `GetProcAddress` dozens or hundreds of times in a loop, it's likely resolving imports for a manually mapped DLL. This pattern is very different from normal import resolution, which happens automatically during DLL loading.

**Relocation fixing** involves code that reads a relocation table and modifies memory based on the base address. You'll see code that reads data from one location (the relocation table), calculates offsets, and writes to other locations (the code/data being relocated). This often involves pointer arithmetic and memory writes in a loop, which is characteristic of relocation processing.

**No LoadLibrary** is the defining characteristic. If you enumerate loaded modules using tools like Process Hacker, Process Explorer, or the Windows API (`EnumProcessModules`), manually mapped DLLs won't appear in the list. The Windows loader doesn't know about them because they weren't loaded through `LoadLibrary`. However, the DLL is still in memory and executing code.

### Finding Manually Mapped DLLs

Locating manually mapped DLLs requires a combination of dynamic analysis, memory scanning, and pattern recognition.

**Monitor memory allocations** using tools like Process Monitor (ProcMon) or API Monitor. Set filters to watch for `VirtualAlloc`, `VirtualAllocEx`, `NtAllocateVirtualMemory`, and similar functions. Look for large allocations with execute permissions. When you see such an allocation, note the address and size, then examine that memory region in a debugger or memory viewer.

**Look for suspicious code** that performs manual mapping operations. Set breakpoints on `GetProcAddress` and `GetModuleHandle` in your debugger. If these functions are called many times in quick succession, examine the call stack to see what code is calling them. This code is likely performing import resolution for a manually mapped DLL. You can then trace back to find where the DLL was loaded.

**Analyze memory** by scanning for PE headers in allocated regions. Manually mapped DLLs still have PE headers (they need them for the mapping process), so you can scan memory for the "MZ" signature (0x4D5A) followed by a valid PE structure. Tools like PE-sieve or custom scripts can automate this process. When you find a PE header in an allocated region that's not in the module list, you've likely found a manually mapped DLL.

**Use debugger** techniques like setting breakpoints on `VirtualAlloc` and tracing execution. When `VirtualAlloc` is called with execute permissions, let it return, then set a memory breakpoint on the allocated region. When code writes to that region (copying the DLL), your breakpoint will trigger. You can then trace the execution to see the entire manual mapping process, including import resolution and relocation fixing.

## Analyzing Manually Mapped DLLs

Once you've identified a manually mapped DLL, you need to extract and analyze it to understand what it does.

### Steps

The analysis process involves locating the DLL in memory, extracting it, and then analyzing it with your normal reverse engineering tools.

**Find the DLL in memory** by identifying its base address. If you found it through memory scanning, you already have the address. If you found it through monitoring allocations, the allocation address is the base address. You can verify it's a DLL by checking for the "MZ" signature and valid PE headers at that address.

**Dump the DLL** by extracting it from memory to a file. Use your debugger's memory dump feature, a tool like Scylla or PE-sieve, or write a custom script. You need to dump the entire DLL, from the base address to base address + SizeOfImage (found in the PE optional header). This gives you a complete copy of the DLL as it exists in memory.

**Fix the DLL** if necessary. Sometimes manually mapped DLLs have modified or corrupted PE headers to evade detection. You may need to rebuild the import table, fix section headers, or repair other PE structures. Tools like PE-bear can help you examine and fix the PE structure. If the IAT has been resolved, you might want to rebuild it to show the original import names rather than just addresses.

**Analyze the DLL** using Binary Ninja, Ghidra, or IDA Pro. Once you have a valid PE file, you can analyze it like any other DLL. Look for the `DllMain` function to see what the DLL does when loaded. Examine exported functions to understand the DLL's interface. Analyze the code to determine the DLL's purpose—is it a game cheat, malware, a rootkit, or something else?

### Dumping the DLL

The dumping process requires careful attention to ensure you get a complete and valid copy of the DLL.

**Find the base address** by using the techniques described earlier—memory scanning, allocation monitoring, or debugger breakpoints. The base address is where the PE headers start (the "MZ" signature). Verify you have the correct address by checking that the PE headers are valid.

**Find the size** by reading the `SizeOfImage` field from the PE optional header. This field is at offset 0x50 in the optional header (which starts after the COFF header). The size tells you how many bytes to dump. Don't just dump a fixed size—different DLLs have different sizes, and dumping too little will give you an incomplete DLL.

**Dump the memory** using your tool of choice. In x64dbg, right-click in the memory view, select "Dump Memory to File", and specify the base address and size. In WinDbg, use the `.writemem` command. In Python with the `ctypes` or `pymem` library, use `ReadProcessMemory`. Make sure you dump the entire region from base address to base address + SizeOfImage.

**Save to file** with a `.dll` extension. The dumped file should be a valid PE file that you can open in PE analysis tools. Verify the dump by opening it in PE-bear or CFF Explorer—you should see valid PE headers, sections, and import/export tables. If the file doesn't look right, you may have dumped the wrong address or size, or the DLL's headers may be corrupted.

## Exercises

### Exercise 1: Identify Manually Mapped DLLs

**Objective**: Learn to identify manually mapped DLLs.

**Steps**:
1. Create a program that manually maps a DLL
2. Run it in a debugger
3. Identify the manually mapped DLL
4. Document your findings

**Verification**: You should be able to identify the DLL.

### Exercise 2: Dump a Manually Mapped DLL

**Objective**: Learn to dump manually mapped DLLs.

**Steps**:
1. Create a program that manually maps a DLL
2. Run it in a debugger
3. Find the DLL in memory
4. Dump the DLL to a file
5. Verify the dumped DLL is valid

**Verification**: The dumped DLL should be valid.

### Exercise 3: Analyze a Manually Mapped DLL

**Objective**: Learn to analyze manually mapped DLLs.

**Steps**:
1. Dump a manually mapped DLL
2. Open it in Binary Ninja
3. Analyze the DLL
4. Understand what it does

**Verification**: You should understand the DLL's functionality.

## Solutions

### Solution 1: Identify Manually Mapped DLLs

Manually mapped DLLs typically:
- Are allocated with VirtualAlloc
- Have PE headers in memory
- Are not in the module list
- Have manual import resolution

### Solution 2: Dump a Manually Mapped DLL

To dump a manually mapped DLL:
1. Find the base address
2. Find the size
3. Dump the memory
4. Save to a file

### Solution 3: Analyze a Manually Mapped DLL

To analyze a manually mapped DLL:
1. Dump the DLL
2. Open in Binary Ninja
3. Analyze the code
4. Understand the functionality

## Summary

You now understand manually mapped DLLs. You can:

- Identify manually mapped DLLs
- Dump manually mapped DLLs
- Analyze manually mapped DLLs
- Understand injection techniques
- Detect and analyze injected code

In the next lesson, you'll learn about PDBs and symbols.
