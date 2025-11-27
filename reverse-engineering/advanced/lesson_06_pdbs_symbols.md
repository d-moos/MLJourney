# Lesson 6: PDBs and Symbols – Using Debug Information

## Overview

PDB files (Program Database) contain debug information that makes reverse engineering much easier. They include:
- Function names and addresses
- Variable names and types
- Source file information
- Line number information

Understanding PDBs helps you:
- Analyze binaries with debug information
- Use symbol servers to download PDBs
- Understand how debug information is stored
- Recognize when PDBs are available

## What You'll Learn

By the end of this lesson, you will understand:

- PDB file format
- How to load PDBs in analysis tools
- How to use symbol servers
- How to extract information from PDBs
- How to use PDB information for analysis

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-5 of this course
- Understand PE file structure
- Be comfortable with Binary Ninja and x64dbg

## PDB File Format

PDB (Program Database) files are Microsoft's proprietary format for storing debugging information. Understanding the PDB format helps you extract maximum value from debug symbols and even work with corrupted or partial PDB files.

PDB files are structured as a multi-stream database, similar to a file system within a file. Each stream contains a different type of information, and the streams are indexed for efficient access.

**Streams** are the fundamental organizational unit in PDB files. Each stream is a sequence of bytes containing related information. The PDB format uses a stream directory to locate and access individual streams.

**Stream 0** is the old directory format, now deprecated but still present for backward compatibility. Modern tools ignore this stream and use the newer directory format stored elsewhere in the file.

**Stream 1** contains PDB information including the PDB version, signature (GUID), age (incremented each time the PDB is updated), and the names of other streams. This stream is essential for matching PDB files to binaries—the binary's debug directory contains a GUID and age that must match the PDB's GUID and age.

**Stream 2** contains type information (TPI - Type Information) including definitions of all types used in the program: structures, classes, enums, typedefs, and function signatures. This information is crucial for understanding data structures and function prototypes.

**Stream 3** contains debug information (DBI - Debug Information) including module information, section contributions, source file information, and the mapping between addresses and source lines. This is the stream that tells you which source file and line number corresponds to each address in the binary.

**Stream 4+** contain module-specific information. Each compiled object file (`.obj`) becomes a module in the PDB, and each module has its own stream containing symbols (function names, variable names, etc.) specific to that module. Large programs can have hundreds of module streams.

## Loading PDBs in Analysis Tools

Loading PDB files dramatically improves your reverse engineering workflow by providing function names, variable names, and type information. Each tool has its own method for loading PDBs.

### Binary Ninja

Binary Ninja has excellent PDB support with automatic loading and symbol resolution.

To use PDBs in Binary Ninja, place the PDB file in the same directory as the binary you're analyzing. The PDB must have the same base name as the binary (e.g., `program.exe` and `program.pdb`).

Open the binary in Binary Ninja, and the tool automatically detects and loads the PDB file. You'll see a notification in the log window indicating that symbols are being loaded.

Once loaded, function names and types are displayed throughout the analysis. Functions that were previously named `sub_140001000` now show their real names like `ProcessUserInput`. Variables have meaningful names instead of generic names like `var_10`. Type information is applied to function parameters and return values, making the decompiler output much more readable.

### x64dbg

x64dbg is a powerful debugger that can load PDB files to enhance your debugging experience with symbol information.

To load a PDB in x64dbg, go to File → Load PDB (or use the Symbols tab). Select the PDB file you want to load. x64dbg parses the PDB and loads all the debug information.

Once loaded, function names are displayed in the disassembly view. Instead of seeing addresses like `0x140001000`, you see function names like `ProcessUserInput`. This makes it much easier to understand what code you're looking at.

The symbols are also available in the Symbols tab, where you can search for specific functions or variables by name. You can set breakpoints on functions by name (like `bp ProcessUserInput`) instead of having to find the address manually.

### Ghidra

Ghidra has robust PDB support through its built-in PDB analyzer.

To load a PDB in Ghidra, open your binary, then go to File → Load PDB File. Select the PDB file you want to load. Ghidra parses the PDB and applies the debug information to your analysis.

Ghidra's PDB loader extracts function names, variable names, type information, and even source file information. Functions are renamed, types are applied, and the decompiler output becomes significantly more readable. Ghidra can also use the type information to improve its analysis, identifying function parameters and return types more accurately.

## Symbol Servers

Symbol servers are HTTP-based repositories that store PDB files for public distribution. They allow you to automatically download the correct PDB file for any binary without having to manually search for it.

Microsoft maintains a public symbol server that hosts PDB files for all Windows system binaries (like `ntdll.dll`, `kernel32.dll`, etc.). This is invaluable for reverse engineering because you can get full symbol information for Windows APIs.

### Using Microsoft Symbol Server

The Microsoft symbol server is located at `https://msdl.microsoft.com/download/symbols`. This server uses a specific directory structure to organize PDB files based on their GUID and age.

To use the symbol server, configure your analysis tool to point to this URL. Most tools (including WinDbg, x64dbg, and Binary Ninja) have settings for symbol server URLs. Once configured, the tool automatically downloads PDB files as needed.

### Downloading PDBs

The process of downloading a PDB from a symbol server involves constructing the correct URL based on the binary's debug information.

First, get the binary's GUID and age from the PE debug directory. This information is stored in the `IMAGE_DEBUG_DIRECTORY` structure in the PE file. Tools like PE-bear or CFF Explorer can display this information. The GUID is a 128-bit identifier, and the age is a 32-bit counter.

Next, construct the URL using the format: `https://msdl.microsoft.com/download/symbols/binary.pdb/GUID_AGE/binary.pdb`. For example, if you're looking for `ntdll.pdb` with GUID `ABC123...` and age `2`, the URL would be `https://msdl.microsoft.com/download/symbols/ntdll.pdb/ABC1232/ntdll.pdb`.

Finally, download the PDB by accessing the constructed URL. The server returns the compressed PDB file (usually with a `.pd_` extension), which you need to decompress using `expand` on Windows or `cabextract` on Linux.

### Tools for Downloading PDBs

Several tools automate the process of downloading PDB files from symbol servers.

**SymChk** is Microsoft's official symbol checker tool, included in the Windows SDK. It can download PDB files from symbol servers and verify that they match your binaries. Use it with the command `symchk /r binary.exe /s SRV*c:\symbols*https://msdl.microsoft.com/download/symbols` to download symbols to a local cache.

**PDBDownloader** is a third-party tool that provides a simpler interface for downloading PDB files. You provide the binary, and it extracts the GUID/age, constructs the URL, and downloads the PDB automatically.

**Binary Ninja** can automatically download PDBs from symbol servers. Configure the symbol server URL in settings, and Binary Ninja downloads PDB files as needed when you open binaries. This seamless integration makes it easy to always have symbols available.

## Extracting Information from PDBs

Sometimes you need to extract specific information from PDB files programmatically, either for automation or for working with PDBs in custom tools.

### Using pdbparse

`pdbparse` is a Python library for parsing PDB files. It can read PDB files and extract symbols, types, and other debug information. This is useful for writing custom analysis scripts that need to work with symbol information.

You can use `pdbparse` to enumerate all functions in a PDB, extract type definitions, or find specific symbols. The library handles the complex PDB format so you can focus on your analysis logic. However, `pdbparse` is somewhat outdated and may not support the latest PDB format versions.

### Using Ghidra's PDB Analyzer

Ghidra has a built-in PDB analyzer that's more robust than `pdbparse` and supports modern PDB formats.

The analyzer extracts function names and applies them to the binary, renaming functions from generic names like `FUN_140001000` to their real names.

It extracts variable names for local variables, global variables, and function parameters, making the decompiler output much more readable.

It extracts type information including structure definitions, class definitions, enums, and typedefs, and applies this type information throughout the analysis.

It even extracts source file information, showing you which source file and line number corresponds to each function. While you don't have the actual source code, knowing the file and line number can be helpful for understanding the code's organization.

## Exercises

### Exercise 1: Load a PDB in Binary Ninja

Objective: Learn to load PDBs.

Steps:
1. Find a binary with a corresponding PDB file
2. Place the PDB in the same directory as the binary
3. Open the binary in Binary Ninja
4. Verify that function names are displayed
5. Document your findings

Verification: Function names should be displayed.

### Exercise 2: Download a PDB from Symbol Server

Objective: Learn to download PDBs.

Steps:
1. Find a Windows system binary (e.g., kernel32.dll)
2. Get the binary's GUID and age
3. Download the PDB from Microsoft's symbol server
4. Load the PDB in Binary Ninja
5. Analyze the binary with debug information

Verification: You should be able to download and load the PDB.

### Exercise 3: Extract Information from a PDB

Objective: Learn to extract PDB information.

Steps:
1. Use pdbparse to parse a PDB file
2. Extract function names and addresses
3. Extract variable names and types
4. Create a report of the extracted information

Verification: You should be able to extract PDB information.

## Solutions

### Solution 1: Load a PDB in Binary Ninja

To load a PDB:
1. Place the PDB in the same directory as the binary
2. Open the binary in Binary Ninja
3. Binary Ninja automatically loads the PDB
4. Function names are displayed

### Solution 2: Download a PDB from Symbol Server

To download a PDB:
1. Get the binary's GUID and age
2. Construct the URL
3. Download the PDB
4. Load in Binary Ninja

### Solution 3: Extract Information from a PDB

To extract PDB information:
1. Use pdbparse to parse the PDB
2. Iterate through streams
3. Extract function names, addresses, and types
4. Create a report

## Summary

You now understand PDBs and symbols. You can:

- Load PDBs in analysis tools
- Download PDBs from symbol servers
- Extract information from PDBs
- Use debug information for analysis
- Understand PDB file format

In the next lesson, you'll learn about custom tools.
