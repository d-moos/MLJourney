# Lesson 3: PE Format Basics – How Windows Loads Binaries

## Overview

Every Windows executable and DLL follows the **Portable Executable (PE)** format. This format is a container that tells Windows how to load and run the binary. It specifies where the code is, where the data is, which DLLs to load, which functions to import, and much more.

Understanding PE structure is critical for reverse engineering because it helps you navigate a binary. When you open a binary in Binary Ninja, you're looking at the code and data extracted from the PE file. When you use PE-bear to inspect a binary, you're examining the PE structure directly. This lesson teaches you to read and understand PE files.

## What You'll Learn

By the end of this lesson, you will understand:

- **The PE file structure**: How headers and sections are organized
- **The DOS header and PE signature**: Legacy compatibility and the PE marker
- **The COFF header**: Machine type, number of sections, and other metadata
- **The optional header**: Image base, entry point, and subsystem information
- **Sections**: What `.text`, `.rdata`, `.data`, and `.rsrc` contain
- **The import table**: How Windows knows which DLLs to load and which functions to import
- **RVA vs file offset**: How to convert between addresses in memory and offsets in the file
- **The entry point**: Where execution starts when the binary runs

## Prerequisites

Before starting this lesson, you should:

- Have completed Lesson 1 (lab setup)
- Be comfortable with hexadecimal notation (0x1000, 0x400, etc.)
- Have a basic understanding of memory addresses

## The PE File Structure

A PE file is organized into several parts:

1. **DOS Header** (64 bytes): Legacy DOS compatibility header
2. **DOS Stub** (variable): Legacy DOS program (usually just prints "This program cannot be run in DOS mode")
3. **PE Signature** (4 bytes): The magic bytes "PE\0\0" (0x50450000)
4. **COFF Header** (20 bytes): Machine type, number of sections, timestamp, etc.
5. **Optional Header** (224 bytes for x64): Image base, entry point, subsystem, etc.
6. **Section Headers** (40 bytes each): Describes each section (name, size, location, etc.)
7. **Sections**: The actual code and data (`.text`, `.rdata`, `.data`, `.rsrc`, etc.)

Let's examine each part in detail.

## The DOS Header

The DOS header is a legacy artifact from the days when Windows binaries needed to be compatible with DOS. It starts with the magic bytes "MZ" (0x4D5A), which stands for Mark Zbikowski (one of the original DOS developers).

The DOS header contains:

- **Offset 0x00**: Magic bytes "MZ" (0x4D5A)
- **Offset 0x3C**: Offset to the PE signature (usually 0x40 or 0x80)

The rest of the DOS header is mostly unused in modern PE files. The important thing is that the DOS header tells you where the PE signature is located.

## The PE Signature and COFF Header

At the offset specified in the DOS header (usually 0x40), you'll find the PE signature: the bytes "PE\0\0" (0x50450000).

Immediately after the PE signature is the **COFF header** (20 bytes):

- **Offset 0x00-0x01**: Machine type (0x8664 for x64, 0x014C for x86)
- **Offset 0x02-0x03**: Number of sections
- **Offset 0x04-0x07**: Timestamp (when the binary was compiled)
- **Offset 0x08-0x0B**: Pointer to symbol table (usually 0 for modern binaries)
- **Offset 0x0C-0x0F**: Number of symbols (usually 0)
- **Offset 0x10-0x11**: Size of optional header (224 for x64, 96 for x86)
- **Offset 0x12-0x13**: Characteristics (flags indicating executable, DLL, etc.)

The most important fields are:
- **Machine type**: Tells you if it's x86 or x64
- **Number of sections**: How many sections the binary has
- **Characteristics**: Indicates if it's an executable, DLL, or other type

## The Optional Header

The optional header (despite its name, it's required for executables) contains critical information about how to load and run the binary:

- **Magic number** (0x20B for x64, 0x10B for x86): Confirms this is a PE file
- **Entry point RVA**: The address (relative to the image base) where execution starts
- **Image base**: The preferred base address where the binary is loaded into memory (usually 0x400000 for executables, 0x180000000 for x64 DLLs)
- **Section alignment**: How sections are aligned in memory (usually 0x1000 = 4 KB)
- **File alignment**: How sections are aligned in the file (usually 0x200 = 512 bytes)
- **Subsystem**: Indicates if it's a console app, GUI app, driver, etc.
- **Size of image**: Total size of the binary when loaded into memory
- **Size of headers**: Size of all headers combined

The entry point RVA is particularly important: it tells you where execution starts. When you open a binary in Binary Ninja, the entry point is usually highlighted or marked as the start of execution.

## Sections

A section is a contiguous block of code or data within the PE file. Each section has a header (40 bytes) that describes it:

- **Name** (8 bytes): Usually `.text`, `.data`, `.rdata`, `.rsrc`, etc.
- **Virtual size**: Size of the section when loaded into memory
- **Virtual address (RVA)**: Address of the section when loaded into memory
- **Size of raw data**: Size of the section in the file
- **Pointer to raw data**: Offset of the section in the file
- **Characteristics**: Flags indicating if the section is executable, readable, writable, etc.

### Common Sections

**`.text`** contains executable code. It's marked as readable and executable, but not writable.

**`.rdata`** (read-only data) contains constants, strings, and other read-only data. It's marked as readable but not writable or executable.

**`.data`** contains initialized global variables. It's marked as readable and writable.

**`.bss`** (or `.data` with virtual size > raw size) contains uninitialized global variables. It's marked as readable and writable.

**`.rsrc`** contains resources like icons, dialogs, and version information. It's marked as readable but not writable or executable.

**`.reloc`** contains relocation information used when the binary is loaded at a different address than the image base.

When you open a binary in Binary Ninja, each section is displayed separately. The `.text` section contains the code you'll be analyzing.

## RVA vs File Offset

This is a critical concept: **RVA** (Relative Virtual Address) is an address relative to the image base when the binary is loaded into memory. **File offset** is a byte offset within the PE file on disk.

When Windows loads a binary, it:
1. Allocates memory at the image base address
2. Copies each section from the file to its RVA in memory
3. Jumps to the entry point RVA

So the same data has two different addresses:
- **In the file**: File offset (e.g., 0x1000)
- **In memory**: RVA (e.g., 0x2000)

To convert between them, you need to know which section the address is in:

**File offset to RVA**:
```
RVA = File offset - Section.PointerToRawData + Section.VirtualAddress
```

**RVA to file offset**:
```
File offset = RVA - Section.VirtualAddress + Section.PointerToRawData
```

### Example

Suppose you have:
- Image base: 0x400000
- `.text` section:
  - Virtual address (RVA): 0x1000
  - Pointer to raw data (file offset): 0x400
  - Size: 0x1000

If you see an address 0x401000 in memory, that's:
- Image base (0x400000) + RVA (0x1000) = 0x401000

To find this in the file:
- File offset = 0x1000 - 0x1000 + 0x400 = 0x400

So the address 0x401000 in memory corresponds to file offset 0x400 in the file.

## The Import Table

The import table tells Windows which DLLs to load and which functions to import from them. It's stored in the `.rdata` section and consists of:

1. **Import Directory Table**: An array of import descriptors, one per DLL
2. **Import Name Table (INT)**: For each DLL, an array of function names/ordinals
3. **Import Address Table (IAT)**: For each DLL, an array of function addresses (filled in by Windows at load time)

When Windows loads a binary:
1. It reads the import directory table
2. For each DLL, it loads the DLL into memory
3. For each function in the INT, it looks up the function in the DLL
4. It writes the function's address into the corresponding IAT entry

When the binary calls an imported function, it actually jumps to the address in the IAT.

### Example

Suppose your binary imports `GetStdHandle` from `kernel32.dll`. The import table contains:

```
Import Directory:
  DLL name: "kernel32.dll"
  INT: [GetStdHandle, WriteFile, ReadFile, ...]
  IAT: [0x00000000, 0x00000000, 0x00000000, ...]  (initially empty)

After Windows loads the binary:
  IAT: [0x7FFF1234, 0x7FFF5678, 0x7FFF9ABC, ...]  (filled with function addresses)
```

When your code calls `GetStdHandle`, it actually does:
```
call [IAT + 0]  ; Call the address in the first IAT entry
```

This is why the IAT is so important in reverse engineering: it's where imported functions are called from.

## Examining PE Files with PE-bear

PE-bear is a tool for inspecting PE files. Let's use it to examine a binary:

1. Open PE-bear
2. File → Open → select your binary
3. You'll see several tabs:
   - **DOS Header**: Shows the DOS header fields
   - **PE Header**: Shows the COFF header and optional header
   - **Sections**: Shows all sections with their properties
   - **Imports**: Shows imported DLLs and functions
   - **Exports**: Shows exported functions (if any)
   - **Resources**: Shows resources (if any)

In the Sections tab, you can see:
- Section name (`.text`, `.data`, etc.)
- Virtual address (RVA)
- Virtual size
- Raw size
- File offset
- Characteristics (executable, readable, writable)

In the Imports tab, you can see:
- DLL name
- Functions imported from that DLL
- Ordinal (if imported by ordinal instead of name)

## Connecting PE Concepts to Binary Ninja

When you open a binary in Binary Ninja, you're looking at the code extracted from the PE file. Here's how PE concepts map to Binary Ninja:

- **Sections**: Binary Ninja shows sections in the left panel. You can click on a section to view its contents.
- **Entry point**: Binary Ninja highlights the entry point function (usually called `entry` or `main`).
- **Imports**: Binary Ninja shows imported functions in the function list. When you see a function call to an imported function, Binary Ninja shows the DLL and function name.
- **Strings**: Binary Ninja shows strings extracted from the `.rdata` section.

When you see an address in Binary Ninja (e.g., 0x401000), that's an RVA + image base. Binary Ninja automatically handles the conversion to file offsets when needed.

## Exercises

### Exercise 1: Section Survey

**Objective**: Understand the structure of sections in a PE file.

**Steps**:
1. Open your `hello_reversing.exe` from Lesson 1 in PE-bear
2. Click on the "Sections" tab
3. For each section, note:
   - Section name
   - Virtual address (RVA)
   - Virtual size
   - Raw size
   - File offset
   - Characteristics (executable, readable, writable)
4. Create a table documenting all sections

**Verification**: You should see sections like:
- `.text`: Executable, readable (contains code)
- `.rdata`: Readable (contains read-only data and strings)
- `.data`: Readable, writable (contains initialized data)
- `.reloc`: Readable (contains relocation information)

### Exercise 2: Entry Point and RVA Mapping

**Objective**: Understand how to convert between RVA and file offset.

**Steps**:
1. In PE-bear, go to the "PE Header" tab
2. Find the "Entry point RVA" field
3. Note the value (e.g., 0x1000)
4. Find which section contains this RVA
5. Calculate the file offset using the formula:
   ```
   File offset = RVA - Section.VirtualAddress + Section.PointerToRawData
   ```
6. Open the binary in a hex editor and navigate to that file offset
7. Verify that you see the start of the `.text` section
8. Open the binary in Binary Ninja and confirm that the entry point matches

**Verification**: You should be able to:
- Find the entry point RVA in PE-bear
- Calculate the corresponding file offset
- Verify the calculation by examining the binary in a hex editor
- Confirm that Binary Ninja's entry point matches

### Exercise 3: Import Analysis

**Objective**: Understand how imports work and identify common imported functions.

**Steps**:
1. In PE-bear, click on the "Imports" tab
2. Expand each DLL to see the imported functions
3. For each DLL, note:
   - DLL name
   - Number of functions imported
   - Names of at least 3 functions
4. For each function, look up its documentation (search online or use Windows API documentation)
5. Create a document describing what each function does

**Verification**: You should identify imports like:
- `kernel32.dll`: Core Windows API
  - `GetStdHandle`: Get a handle to standard input/output
  - `WriteFile`: Write data to a file or console
  - `ReadFile`: Read data from a file or console
  - `ExitProcess`: Exit the program
- `ntdll.dll`: Native API (used internally)

## Solutions

### Solution 1: Section Survey

When you open `hello_reversing.exe` in PE-bear and examine the sections, you should see something like:

| Section | RVA | Virtual Size | Raw Size | File Offset | Characteristics |
|---------|-----|--------------|----------|-------------|-----------------|
| `.text` | 0x1000 | 0x2000 | 0x2000 | 0x400 | Executable, Readable |
| `.rdata` | 0x3000 | 0x1000 | 0x1000 | 0x2400 | Readable |
| `.data` | 0x4000 | 0x1000 | 0x1000 | 0x3400 | Readable, Writable |
| `.reloc` | 0x5000 | 0x1000 | 0x1000 | 0x4400 | Readable |

**Key observations**:
- `.text` is executable and contains code
- `.rdata` is readable but not writable (read-only data)
- `.data` is readable and writable (initialized data)
- `.reloc` contains relocation information

### Solution 2: Entry Point and RVA Mapping

In PE-bear, you'll find the entry point RVA in the "PE Header" tab. For a typical Rust binary, it might be 0x1000.

To convert to file offset:
1. Find the section containing RVA 0x1000 (usually `.text`)
2. `.text` has:
   - Virtual address: 0x1000
   - Pointer to raw data: 0x400
3. Calculate: File offset = 0x1000 - 0x1000 + 0x400 = 0x400

So the entry point is at file offset 0x400 in the binary.

When you open the binary in a hex editor and navigate to offset 0x400, you should see the start of the `.text` section (usually some assembly instructions).

When you open the binary in Binary Ninja, the entry point function should be highlighted, and its address should match the entry point RVA.

### Solution 3: Import Analysis

When you examine the imports in PE-bear, you should see:

**kernel32.dll**:
- `GetStdHandle`: Returns a handle to standard input, output, or error
- `WriteFile`: Writes data to a file or device (including console)
- `ReadFile`: Reads data from a file or device (including console)
- `ExitProcess`: Terminates the current process
- `GetLastError`: Returns the error code of the last failed function

**ntdll.dll**:
- `RtlUserThreadStart`: Internal function for thread startup
- `NtQueryInformationProcess`: Query process information

These are typical imports for a simple console application. More complex applications might import additional DLLs like `user32.dll` (GUI), `advapi32.dll` (security), etc.

## Summary

You now understand the PE file format and how Windows loads binaries. You know:

- The structure of PE files (headers, sections, etc.)
- How to identify sections and understand their purpose
- How to convert between RVA and file offset
- How the import table works
- How to use PE-bear to inspect binaries
- How PE concepts map to what you see in Binary Ninja

In the next lesson, you'll learn to use Binary Ninja for static analysis—reading and understanding code without running it.
