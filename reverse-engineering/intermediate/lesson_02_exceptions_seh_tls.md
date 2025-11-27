# Lesson 2: Exceptions, SEH, and TLS – Advanced Control Flow

## Overview

Windows has several mechanisms for handling exceptional conditions and thread-local data:

- **Structured Exception Handling (SEH)**: A mechanism for handling exceptions (like divide by zero)
- **TLS Callbacks**: Functions that run before the main entry point
- **Vectored Exception Handlers**: Global exception handlers

Understanding these is important because:
- Malware uses them to hide code
- Anti-debugging code uses them to detect debuggers
- Packers use them to unpack code
- Game cheats use them to hook functions

## What You'll Learn

By the end of this lesson, you will understand:

- **How SEH works** and how to recognize it in disassembly
- **TLS callbacks** and when they execute
- **Vectored exception handlers** and how to use them
- **How to recognize SEH-based anti-debugging**
- **How to bypass SEH-based protections**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lesson 1 of this course
- Understand x86-64 assembly
- Be comfortable with the Windows API

## Structured Exception Handling (SEH)

SEH is a mechanism for handling exceptions. When an exception occurs (like a divide by zero), Windows looks for an exception handler to handle it.

### SEH Chain

Each thread has a chain of exception handlers. When an exception occurs, Windows walks the chain looking for a handler.

### SEH in Assembly

In assembly, SEH is typically implemented using `try`/`except` blocks:

```c
__try {
    // Code that might throw an exception
    int x = 10 / 0;  // Divide by zero
} __except (EXCEPTION_EXECUTE_HANDLER) {
    // Exception handler
    printf("Exception caught!");
}
```

This compiles to assembly that sets up an exception handler frame.

### Recognizing SEH in Disassembly

When you're reverse engineering a binary and encounter SEH, you'll see characteristic assembly patterns that set up the exception handling infrastructure. Understanding these patterns helps you identify where exception handlers are installed and what they do.

In disassembly, SEH setup typically involves several steps. First, you'll see a `push` instruction that pushes the address of an exception handler function onto the stack. This is the function that will be called if an exception occurs. Next, you'll see another `push` that pushes the address of the previous exception handler in the chain—this maintains the linked list of handlers. Finally, you'll see a `mov` instruction that updates the FS (on x86) or GS (on x64) segment register to point to the new exception handler frame, effectively adding it to the chain.

The pattern typically looks like this in x86 assembly:
```asm
push offset exception_handler    ; Push handler address
push fs:[0]                       ; Push previous handler
mov fs:[0], esp                   ; Set new handler as current
```

On x64, the pattern is different because x64 uses table-based exception handling rather than frame-based SEH, but you'll still see references to exception handler tables in the PE file's exception directory.

## TLS Callbacks

**TLS (Thread Local Storage) callbacks** are one of the most important—and often overlooked—features in Windows PE files. They're functions that execute before the main entry point, making them perfect for initialization code, anti-debugging tricks, and unpacking routines. Many reverse engineers miss TLS callbacks because they focus on the entry point, not realizing that code has already executed before they even reach `main`.

TLS callbacks are particularly useful for several purposes. They're designed for initializing thread-local data—variables that each thread has its own copy of. However, malware authors have discovered that TLS callbacks are perfect for running anti-debugging code before the debugger reaches the entry point. If you set a breakpoint at the entry point, the anti-debugging code in the TLS callback has already run, potentially detecting your debugger and altering the program's behavior. Packers also use TLS callbacks to unpack code before the main program runs, making it harder to find the unpacking routine.

### How TLS Callbacks Work

The TLS callback mechanism is built into the Windows PE loader and operates automatically when a process or thread is created. Understanding the execution flow is crucial for effective reverse engineering.

The process begins with the PE file containing a TLS directory in its optional header. This directory is one of the data directories (like the import directory or export directory) and contains information about thread-local storage. Within the TLS directory is a pointer to an array of callback function addresses. This array is null-terminated, meaning it ends with a zero pointer.

When Windows creates a new thread (including the initial thread when the process starts), it checks if the PE file has a TLS directory. If it does, Windows walks through the array of TLS callback addresses and calls each callback function in order. Each callback is called with parameters indicating why it's being called (thread attach, thread detach, process attach, or process detach), similar to `DllMain`. Only after all TLS callbacks have completed does Windows transfer control to the main entry point specified in the PE header.

This means TLS callbacks execute before your debugger's entry point breakpoint, before any initialization code you might expect, and before any anti-anti-debugging plugins have a chance to patch the binary. This makes them extremely powerful for both legitimate initialization and malicious anti-analysis.

### Recognizing TLS Callbacks

Identifying TLS callbacks requires examining the PE file structure, which you can do in Binary Ninja or with PE analysis tools like PE-bear.

In Binary Ninja, start by examining the PE header. Navigate to the optional header's data directories section and look for the TLS directory entry. If the TLS directory's RVA (Relative Virtual Address) is non-zero, the binary has TLS data and potentially TLS callbacks.

Follow the TLS directory RVA to the TLS directory structure itself. This structure contains several fields, but the most important is the `AddressOfCallBacks` field. This field contains a pointer (an absolute virtual address, not an RVA) to an array of callback function pointers.

Navigate to the address specified in `AddressOfCallBacks`. You'll see an array of 8-byte pointers (on x64) or 4-byte pointers (on x86). Each non-zero pointer is the address of a TLS callback function. The array ends with a zero pointer. Click on each callback address to jump to the callback function and analyze what it does. These functions run before the main entry point, so any anti-debugging or unpacking code here executes first.

### Example TLS Callback

```c
void __stdcall tls_callback(PVOID DllHandle, DWORD Reason, PVOID Reserved) {
    if (Reason == DLL_THREAD_ATTACH) {
        // Thread is being created
        printf("Thread created!");
    }
}
```

## Vectored Exception Handlers

Vectored exception handlers are global exception handlers that are called for all exceptions in a process.

### Adding a Vectored Exception Handler

```c
LONG WINAPI VectoredExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo) {
    // Handle the exception
    return EXCEPTION_CONTINUE_EXECUTION;
}

// Add the handler
AddVectoredExceptionHandler(1, VectoredExceptionHandler);
```

### Using Vectored Exception Handlers for Anti-Debugging

Some anti-debugging code uses vectored exception handlers to detect debuggers:

```c
LONG WINAPI VectoredExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT) {
        // Breakpoint detected!
        exit(1);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
```

## Exercises

### Exercise 1: Recognize SEH in Disassembly

**Objective**: Learn to identify SEH in disassembly.

**Steps**:
1. Write a C program with a try/except block
2. Compile it
3. Open it in Binary Ninja
4. Find the try/except block in the disassembly
5. Identify the exception handler setup
6. Document your findings

**Verification**: You should be able to identify SEH structures in disassembly.

### Exercise 2: Find TLS Callbacks

**Objective**: Learn to find and analyze TLS callbacks.

**Steps**:
1. Create a binary with a TLS callback
2. Open it in Binary Ninja
3. Look at the PE header
4. Find the TLS directory
5. Find the TLS callback function
6. Analyze what the callback does

**Verification**: You should be able to locate and analyze TLS callbacks.

### Exercise 3: Implement Anti-Debugging with Vectored Exception Handlers

**Objective**: Learn to use vectored exception handlers.

**Steps**:
1. Write a program that:
   - Adds a vectored exception handler
   - The handler detects breakpoints
   - If a breakpoint is detected, exit
2. Compile and run it normally (should work)
3. Run it in a debugger and set a breakpoint (should exit)

**Verification**: The program should detect breakpoints and exit.

## Solutions

### Solution 1: Recognize SEH in Disassembly

When you look at a try/except block in disassembly, you'll see:
- A `push` of an exception handler address
- A `push` of the previous exception handler
- A `mov` to set up the exception handler chain
- The exception handler code

### Solution 2: Find TLS Callbacks

When you look at the PE header in Binary Ninja:
1. Find the TLS directory
2. The TLS directory contains a list of callback addresses
3. Each callback is a function that runs before main
4. You can analyze each callback to see what it does

### Solution 3: Implement Anti-Debugging with Vectored Exception Handlers

A simple anti-debugging program:

```c
LONG WINAPI VectoredExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT) {
        exit(1);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

int main() {
    AddVectoredExceptionHandler(1, VectoredExceptionHandler);
    // ... rest of program
}
```

## Summary

You now understand advanced control flow mechanisms. You can:

- Recognize and analyze SEH
- Find and analyze TLS callbacks
- Implement vectored exception handlers
- Recognize anti-debugging techniques using these mechanisms

In the next lesson, you'll learn about packers and manual unpacking.
