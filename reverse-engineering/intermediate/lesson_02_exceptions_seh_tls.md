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

When you see SEH in disassembly, you'll see:
- A `push` of an exception handler address
- A `push` of the previous exception handler
- A `mov` to set up the exception handler chain

## TLS Callbacks

**TLS callbacks** are functions that run before the main entry point. They're useful for:
- Initializing thread-local data
- Running anti-debugging code
- Unpacking code

### How TLS Callbacks Work

1. The PE file has a TLS directory
2. The TLS directory contains a list of callback functions
3. When a thread is created, Windows calls all TLS callbacks
4. After all TLS callbacks complete, the main entry point is called

### Recognizing TLS Callbacks

In Binary Ninja:
1. Look at the PE header
2. Find the TLS directory
3. The TLS directory contains a list of callback addresses
4. Each callback is a function that runs before main

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
