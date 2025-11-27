# Lesson 4: Binary Ninja Static Analysis – Reading Code Without Running It

## Overview

**Static analysis** means analyzing a binary without running it. You read the code, understand the logic, and figure out what it does. This is the core skill of reverse engineering. In this lesson, you'll learn to use **Binary Ninja**, a professional disassembler and decompiler, to analyze binaries.

Binary Ninja is powerful because it shows you both the raw assembly and a decompiled pseudocode representation. The pseudocode is much easier to read than assembly, but understanding both is important because the decompiler sometimes makes mistakes.

## What You'll Learn

By the end of this lesson, you will understand:

- **How to open and navigate binaries in Binary Ninja**
- **How to read and understand decompiled pseudocode**
- **How to cross-reference functions and data**
- **How to identify and analyze imported functions**
- **How to use Binary Ninja's analysis features** (xrefs, strings, etc.)
- **How to recognize common patterns** in decompiled code
- **How to document your findings** as you analyze

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-3
- Be comfortable with x86-64 assembly (from Lesson 2)
- Understand PE file structure (from Lesson 3)
- Have Binary Ninja installed

## Opening a Binary in Binary Ninja

Let's start by opening your `hello_reversing.exe` from Lesson 1:

1. Open Binary Ninja
2. File → Open → select `hello_reversing.exe`
3. Binary Ninja will analyze the binary (this may take a minute)
4. You'll see the main window with several panels:
   - **Left panel**: Functions list, strings, imports, etc.
   - **Center panel**: Disassembly/pseudocode view
   - **Right panel**: Mini-map and analysis info

## Understanding the Interface

### The Functions List

The left panel shows all functions found in the binary. Functions are organized by:
- **Imported functions**: Functions from DLLs (e.g., `kernel32.WriteFile`)
- **User-defined functions**: Functions defined in the binary

Click on a function to jump to it in the center panel.

### The Disassembly/Pseudocode View

The center panel shows the code. You can toggle between:
- **Disassembly view**: Raw x86-64 assembly instructions
- **Pseudocode view**: Decompiled C-like code (usually easier to read)

To toggle, press `Tab` or click the view selector at the top.

### The Mini-Map

The right panel shows a mini-map of the entire function. You can click on it to jump to different parts of the function.

## Reading Decompiled Pseudocode

Let's look at an example. Suppose Binary Ninja decompiles a function like this:

```c
int64_t main(int64_t arg1, int64_t arg2) {
    int64_t rax = 0;
    
    if (arg1 > 0) {
        rax = arg1 + arg2;
    } else {
        rax = arg1 - arg2;
    }
    
    return rax;
}
```

This is much easier to read than the assembly! You can immediately see:
- The function takes two arguments
- It checks if the first argument is positive
- It either adds or subtracts the arguments
- It returns the result

### Common Pseudocode Patterns

**Variable declarations**: `int64_t rax = 0;` declares a 64-bit integer variable

**Assignments**: `rax = arg1 + arg2;` assigns a value to a variable

**Conditionals**: `if (condition) { ... } else { ... }` represents conditional branches

**Loops**: `while (condition) { ... }` or `for (init; condition; increment) { ... }` represent loops

**Function calls**: `WriteFile(handle, buffer, size);` represents calls to functions

**Return statements**: `return rax;` represents the return value

## Cross-References (Xrefs)

Cross-references show where a function or variable is used. This is crucial for understanding how code flows.

To see cross-references:
1. Right-click on a function or variable name
2. Select "Show xrefs" or press `X`
3. Binary Ninja shows all places where that function/variable is used

For example, if you see `GetStdHandle` is called in 3 places, you can click on each xref to jump to that location.

## Analyzing Imported Functions

Imported functions are called from external DLLs. When you see a call to an imported function, Binary Ninja shows:
- The DLL name (e.g., `kernel32`)
- The function name (e.g., `WriteFile`)
- The function signature (parameters and return type)

Understanding what imported functions do is key to understanding the binary. For example:
- `GetStdHandle(STD_OUTPUT_HANDLE)` gets a handle to the console output
- `WriteFile(handle, buffer, size, ...)` writes data to the console
- `ExitProcess(0)` exits the program

## Strings and Data

Binary Ninja automatically extracts strings from the binary. To see all strings:
1. Click on the "Strings" tab in the left panel
2. You'll see all strings found in the binary
3. Click on a string to jump to where it's used

Strings are often very informative. For example, if you see a string like "Invalid password", you know the binary is checking a password somewhere.

## Exercises

### Exercise 1: Navigate and Understand a Simple Function

**Objective**: Get comfortable with Binary Ninja's interface.

**Steps**:
1. Open `hello_reversing.exe` in Binary Ninja
2. Find the `main` function in the functions list
3. Switch to pseudocode view (press `Tab`)
4. Read the pseudocode and understand what the function does
5. Identify:
   - All function calls
   - All variables
   - All conditionals
6. Document your findings in a text file

**Verification**: You should be able to describe what the `main` function does in plain English.

### Exercise 2: Follow Cross-References

**Objective**: Understand how functions are called.

**Steps**:
1. In Binary Ninja, find an imported function (e.g., `WriteFile`)
2. Right-click on it and select "Show xrefs"
3. For each xref, click on it to jump to that location
4. Understand why that function is being called
5. Document the call sites

**Verification**: You should understand all places where the imported function is called.

### Exercise 3: Analyze Strings

**Objective**: Use strings to understand program behavior.

**Steps**:
1. Click on the "Strings" tab in Binary Ninja
2. Look for interesting strings (error messages, prompts, etc.)
3. For each string, click on it to see where it's used
4. Understand the context of each string
5. Create a document listing all strings and their purposes

**Verification**: You should understand what each string is used for.

## Solutions

### Solution 1: Navigate and Understand a Simple Function

When you open `hello_reversing.exe` and look at the `main` function, you should see pseudocode like:

```c
int64_t main() {
    puts("Welcome to Binary Reversing!");
    puts("Enter your name: ");
    
    char name[256];
    fgets(name, 256, stdin);
    
    printf("Hello, %s!\n", name);
    printf("Your name has %d characters.\n", strlen(name));
    
    return 0;
}
```

**Key observations**:
- The function calls `puts` to print messages
- It calls `fgets` to read user input
- It calls `printf` to print formatted output
- It calls `strlen` to get the length of the string
- It returns 0 (success)

### Solution 2: Follow Cross-References

When you look at xrefs for `WriteFile`, you should see it's called from:
- The `main` function (to write output)
- Possibly other functions

Each xref shows the exact location where the function is called, allowing you to understand the program flow.

### Solution 3: Analyze Strings

When you look at the strings, you should see:
- "Welcome to Binary Reversing!" - Initial greeting
- "Enter your name: " - Prompt for input
- "Hello, %s!" - Greeting with name
- "Your name has %d characters." - Character count message

Each string is used in a specific context, and understanding these contexts helps you understand the program.

## Summary

You now know how to use Binary Ninja for static analysis. You can:

- Open and navigate binaries
- Read decompiled pseudocode
- Follow cross-references
- Analyze imported functions
- Extract and understand strings
- Document your findings

In the next lesson, you'll learn dynamic analysis—running the binary in a debugger to see what it does at runtime.
