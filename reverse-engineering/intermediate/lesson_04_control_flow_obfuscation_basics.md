# Lesson 4: Control Flow Obfuscation Basics – Defeating Code Flattening

## Overview

**Control flow obfuscation** is a technique that makes code harder to understand by obscuring the control flow. Instead of straightforward if/else statements, the code uses complex jumps and conditions.

Common obfuscation techniques include:
- **Code flattening**: Converting if/else into a state machine
- **Opaque predicates**: Conditions that are always true or false
- **Junk code**: Useless code that doesn't affect the program
- **Jump tables**: Using tables to determine where to jump

## What You'll Learn

By the end of this lesson, you will understand:

- **How code flattening works**
- **How to recognize obfuscated code**
- **How to deobfuscate code** (manually and with tools)
- **How to use symbolic execution** to understand obfuscated code
- **How to recognize opaque predicates**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-3 of this course
- Understand x86-64 assembly
- Be comfortable with control flow analysis

## Code Flattening

**Code flattening** converts normal if/else statements into a state machine.

### Example: Original Code

```c
if (x > 10) {
    y = 20;
} else {
    y = 30;
}
z = y + 5;
```

### Example: Flattened Code

```c
int state = 0;
while (true) {
    switch (state) {
        case 0:
            if (x > 10) {
                state = 1;
            } else {
                state = 2;
            }
            break;
        case 1:
            y = 20;
            state = 3;
            break;
        case 2:
            y = 30;
            state = 3;
            break;
        case 3:
            z = y + 5;
            return;
    }
}
```

### Recognizing Flattened Code

In disassembly, flattened code looks like:
- A large switch statement
- Many jumps to different locations
- A state variable that's updated
- Complex control flow

## Opaque Predicates

An **opaque predicate** is a condition that always evaluates to the same value, but it's not obvious.

### Example: Opaque Predicate

```c
if ((x * 2) % 2 == 0) {
    // This is always true
    // (any number times 2 is even)
}
```

### Recognizing Opaque Predicates

In disassembly, opaque predicates look like:
- Complex conditions
- Conditions that always jump or never jump
- Conditions that don't depend on user input

## Junk Code

**Junk code** is code that doesn't affect the program's behavior. It's added to make the code harder to understand.

### Example: Junk Code

```c
int x = 5;
int y = x + 3;  // Junk: y is never used
int z = 10;
```

### Recognizing Junk Code

In disassembly, junk code looks like:
- Variables that are never used
- Calculations that don't affect the result
- Dead code paths

## Deobfuscation Techniques

### Technique 1: Manual Analysis

1. Trace through the code
2. Identify the state variable
3. Map out the state transitions
4. Reconstruct the original control flow

### Technique 2: Symbolic Execution

Symbolic execution is a technique where you execute code symbolically (with variables instead of concrete values) to understand the control flow.

### Technique 3: Automated Deobfuscation

Tools like Ghidra and Binary Ninja have deobfuscation features that can help.

## Exercises

### Exercise 1: Recognize Obfuscated Code

**Objective**: Learn to identify obfuscated code.

**Steps**:
1. Create a simple program with if/else statements
2. Compile it with obfuscation enabled (e.g., using Obfuscator-LLVM)
3. Open it in Binary Ninja
4. Identify the obfuscation techniques used
5. Document your findings

**Verification**: You should be able to identify obfuscation techniques.

### Exercise 2: Deobfuscate Code Manually

**Objective**: Learn to deobfuscate code.

**Steps**:
1. Take an obfuscated binary
2. Trace through the code
3. Identify the state variable
4. Map out the state transitions
5. Reconstruct the original control flow
6. Document your findings

**Verification**: You should be able to reconstruct the original control flow.

### Exercise 3: Recognize Opaque Predicates

**Objective**: Learn to identify opaque predicates.

**Steps**:
1. Create a program with opaque predicates
2. Compile it
3. Open it in Binary Ninja
4. Identify the opaque predicates
5. Determine which branch is always taken
6. Document your findings

**Verification**: You should be able to identify opaque predicates.

## Solutions

### Solution 1: Recognize Obfuscated Code

When you analyze obfuscated code, you should see:
- Complex control flow
- State variables
- Many jumps
- Opaque predicates

### Solution 2: Deobfuscate Code Manually

To deobfuscate code:
1. Identify the state variable
2. Map out the state transitions
3. Reconstruct the original control flow
4. Simplify the code

### Solution 3: Recognize Opaque Predicates

Opaque predicates look like:
- Complex conditions
- Conditions that always jump or never jump
- Conditions that don't depend on user input

## Summary

You now understand control flow obfuscation. You can:

- Recognize obfuscated code
- Identify code flattening
- Recognize opaque predicates
- Deobfuscate code manually
- Use tools to help with deobfuscation

In the next lesson, you'll learn about vulnerability patterns.
