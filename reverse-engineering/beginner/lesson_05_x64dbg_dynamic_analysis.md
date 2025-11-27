# Lesson 5: x64dbg Dynamic Analysis – Running and Debugging Binaries

## Overview

**Dynamic analysis** means running a binary and observing its behavior. You use a **debugger** to step through code, inspect memory and registers, set breakpoints, and modify values at runtime. This is complementary to static analysis—together they give you a complete picture of what a binary does.

**x64dbg** is a free, open-source debugger for Windows x86-64 binaries. It's powerful, user-friendly, and widely used in reverse engineering.

## What You'll Learn

By the end of this lesson, you will understand:

- **How to launch a binary in x64dbg**
- **How to set and use breakpoints**
- **How to step through code** (step into, step over, step out)
- **How to inspect registers and memory**
- **How to modify values at runtime**
- **How to trace function calls**
- **How to use x64dbg's analysis features**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-4
- Be comfortable with x86-64 assembly
- Have x64dbg installed

## Launching a Binary in x64dbg

1. Open x64dbg
2. File → Open → select your binary
3. x64dbg loads the binary and breaks at the entry point
4. You'll see:
   - **Left panel**: Registers, stack, memory
   - **Center panel**: Disassembly
   - **Right panel**: Breakpoints, threads, etc.

## Breakpoints

A **breakpoint** is a marker that tells the debugger to pause execution when it reaches that location. This lets you inspect the state of the program at that point.

### Setting Breakpoints

1. In the disassembly view, right-click on an instruction
2. Select "Breakpoint" or press `F2`
3. The instruction is marked with a red dot
4. When execution reaches that instruction, the debugger pauses

### Conditional Breakpoints

You can set breakpoints that only trigger under certain conditions:
1. Right-click on an instruction
2. Select "Conditional Breakpoint"
3. Enter a condition (e.g., `rax == 5`)
4. The breakpoint only triggers when the condition is true

## Stepping Through Code

Once the debugger is paused, you can step through code:

- **Step Into** (F11): Execute the current instruction. If it's a function call, step into the function.
- **Step Over** (F10): Execute the current instruction. If it's a function call, execute the entire function without stepping into it.
- **Step Out** (Ctrl+Shift+F11): Execute until the current function returns.
- **Run** (F9): Resume execution until the next breakpoint.

## Inspecting Registers and Memory

When the debugger is paused, you can inspect:

### Registers

The left panel shows all registers. You can see:
- **RAX, RBX, RCX, RDX**: General-purpose registers
- **RSP**: Stack pointer
- **RBP**: Base pointer
- **RIP**: Instruction pointer (current instruction)
- **Flags**: Status flags (ZF, CF, SF, OF, etc.)

### Memory

You can view memory at any address:
1. Right-click in the disassembly view
2. Select "Follow in Dump" or press `Ctrl+G`
3. Enter an address
4. x64dbg shows the memory at that address

### The Stack

The stack is displayed in the left panel. You can see:
- Local variables
- Function arguments
- Return addresses

## Modifying Values

You can modify registers and memory at runtime:

### Modifying Registers

1. In the registers panel, right-click on a register
2. Select "Edit"
3. Enter a new value
4. The register is updated

### Modifying Memory

1. In the memory view, right-click on a byte
2. Select "Edit"
3. Enter a new value
4. The memory is updated

This is powerful for testing different code paths without recompiling.

## Tracing Function Calls

x64dbg can trace all function calls:

1. Debug → Trace
2. Select "Trace into" or "Trace over"
3. x64dbg records all instructions executed
4. You can review the trace to see the execution flow

## Exercises

### Exercise 1: Set a Breakpoint and Inspect State

**Objective**: Get comfortable with breakpoints and inspection.

**Steps**:
1. Open `hello_reversing.exe` in x64dbg
2. Find the `main` function
3. Set a breakpoint at the first instruction of `main`
4. Run the program (F9)
5. When the breakpoint is hit, inspect:
   - All registers
   - The stack
   - Local variables
6. Document your findings

**Verification**: You should be able to see the state of the program at the breakpoint.

### Exercise 2: Step Through a Function

**Objective**: Understand how to step through code.

**Steps**:
1. Set a breakpoint at the start of `main`
2. Run the program
3. Step through the function using F10 (step over) and F11 (step into)
4. Observe how registers and memory change
5. Document the execution flow

**Verification**: You should understand how the function executes step-by-step.

### Exercise 3: Modify a Value and Change Behavior

**Objective**: Understand how to modify values at runtime.

**Steps**:
1. Set a breakpoint before a conditional branch
2. Run the program
3. Modify a register or memory value to change the condition
4. Step through the code and observe the different behavior
5. Document how the modification changed the execution

**Verification**: You should see the program take a different code path based on your modification.

## Solutions

### Solution 1: Set a Breakpoint and Inspect State

When you set a breakpoint at the start of `main` and run the program, you should see:

**Registers**:
- RAX: Some value (depends on the binary)
- RBX: Some value
- RCX: First argument to main
- RDX: Second argument to main
- RSP: Stack pointer (points to the top of the stack)
- RBP: Base pointer (points to the base of the stack frame)
- RIP: Address of the first instruction of main

**Stack**:
- The return address (where main will return to)
- Local variables (if any)

### Solution 2: Step Through a Function

When you step through `main`, you should see:
- Each instruction executed
- Registers changing as values are computed
- Memory changing as data is stored
- Function calls being made

### Solution 3: Modify a Value and Change Behavior

If you modify a register before a conditional branch, you can change which branch is taken. For example:
- If you set RAX to 0 before a `cmp rax, 0; je ...` instruction, the jump will be taken
- If you set RAX to 1, the jump won't be taken

This demonstrates that the program's behavior is determined by the values in registers and memory.

## Summary

You now know how to use x64dbg for dynamic analysis. You can:

- Launch binaries in the debugger
- Set and use breakpoints
- Step through code
- Inspect registers and memory
- Modify values at runtime
- Trace function calls

In the next lesson, you'll learn how to patch binaries—modifying them to change their behavior.
