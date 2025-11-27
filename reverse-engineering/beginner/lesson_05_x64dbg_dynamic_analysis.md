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

To begin dynamic analysis, you first need to load your target binary into x64dbg. Launch x64dbg from your Start menu or desktop shortcut—you'll see the main debugger interface appear, initially empty until you load a program.

Go to File → Open and navigate to the binary you want to analyze. For your first session, use the `hello_reversing.exe` you compiled in [Lesson 1: Setting up a Safe Reversing Lab](lesson_01_lab_setup.md). When you select the file and click Open, x64dbg loads the binary into memory and automatically pauses execution at the entry point—the very first instruction that will execute when the program runs.

The x64dbg interface is divided into several panels, each serving a specific purpose. The **left panel** displays registers, the stack, and memory dumps. You'll use this constantly to inspect the current state of the program. The **center panel** shows the disassembly—the assembly instructions that make up the program. This is where you'll spend most of your time, reading through the code and understanding what it does. The **right panel** contains tabs for breakpoints, threads, handles, and other debugging information. Understanding this layout is crucial because you'll be switching between these views constantly as you trace through program execution.

## Breakpoints

A **breakpoint** is a marker that tells the debugger to pause execution when it reaches a specific location in the code. Breakpoints are fundamental to dynamic analysis because they let you stop the program at interesting points and inspect its state—registers, memory, stack contents, and more. Without breakpoints, the program would simply run to completion, and you'd miss all the interesting details of how it works.

### Setting Breakpoints

Setting a basic breakpoint in x64dbg is straightforward. In the disassembly view (the center panel), navigate to the instruction where you want to pause execution. You can scroll through the disassembly or use Ctrl+G to jump to a specific address. Once you've found the instruction of interest, right-click on it and select "Breakpoint" from the context menu, or simply press `F2` as a keyboard shortcut.

The instruction will be marked with a red dot in the left margin, indicating that a breakpoint is active at that location. When you run the program (by pressing F9 or clicking the Run button), execution will proceed normally until it reaches this instruction. At that point, the debugger pauses, and you can inspect the program's state. This is incredibly useful for understanding what's happening at specific points in the code—for example, you might set a breakpoint right before a conditional jump to see what values are being compared.

To remove a breakpoint, simply press F2 again on the same instruction, or right-click and select "Remove Breakpoint". You can also view all active breakpoints in the Breakpoints tab of the right panel, where you can enable, disable, or delete them in bulk.

### Conditional Breakpoints

While basic breakpoints pause execution every time they're reached, conditional breakpoints are more sophisticated—they only trigger when a specific condition is true. This is extremely useful when you're debugging a loop that executes thousands of times, but you only care about a specific iteration, or when you want to catch a rare edge case.

To set a conditional breakpoint, right-click on an instruction and select "Conditional Breakpoint" from the context menu. A dialog will appear where you can enter a condition using x64dbg's expression syntax. For example, you might enter `rax == 5` to break only when the RAX register contains the value 5, or `[rsp] > 0x1000` to break when the value at the top of the stack exceeds a certain threshold.

The condition is evaluated every time execution reaches that instruction. If the condition is true, the debugger pauses; if it's false, execution continues without interruption. This can save you enormous amounts of time compared to manually stepping through code or setting and removing breakpoints repeatedly. You can also combine multiple conditions using logical operators like `&&` (and) and `||` (or), such as `rax == 5 && rbx != 0`.

## Stepping Through Code

Once the debugger is paused at a breakpoint (or at the entry point when you first load a binary), you have several options for controlling execution. These stepping commands are the core of dynamic analysis, allowing you to execute code one instruction at a time and observe exactly how the program behaves.

**Step Into (F11)** executes the current instruction and then pauses again. If the current instruction is a regular operation like `mov` or `add`, it simply executes that instruction and moves to the next one. However, if the current instruction is a function call (`call`), Step Into will follow the call into the function, pausing at the first instruction inside that function. This is useful when you want to understand what a function does internally. For example, if you're at a `call printf` instruction and you Step Into, you'll enter the printf function and see its assembly code.

**Step Over (F10)** also executes the current instruction, but it treats function calls differently. If the current instruction is a `call`, Step Over executes the entire function and pauses at the instruction immediately after the call. This is useful when you're not interested in the internals of a function—for example, if you're analyzing your own code and you encounter a call to a well-known Windows API function like `GetStdHandle`, you probably don't need to step through Microsoft's implementation. Step Over lets you treat the function as a black box.

**Step Out (Ctrl+Shift+F11)** is useful when you've stepped into a function but realize you don't need to see the rest of it. This command executes all remaining instructions in the current function and pauses immediately after the function returns to its caller. It's a quick way to "escape" from a function you've entered.

**Run (F9)** resumes normal execution, allowing the program to run at full speed until it hits another breakpoint, encounters an exception, or terminates. This is how you move between interesting points in the code without manually stepping through every single instruction. The combination of strategic breakpoints and the Run command lets you skip over uninteresting code and focus on the parts that matter.

These stepping commands are the foundation of dynamic analysis. You'll use them constantly as you trace through program execution, and developing an intuition for when to use each one is a key skill in reverse engineering.

## Inspecting Registers and Memory

When the debugger is paused, you have complete visibility into the program's state. This is one of the most powerful aspects of dynamic analysis—you can see exactly what values are in registers, what data is in memory, and how the stack is organized. Understanding how to inspect these elements is crucial for reverse engineering.

### Registers

The left panel of x64dbg displays all CPU registers and their current values. These registers are the CPU's working memory—the fastest storage available, used for calculations, memory addressing, and control flow. Understanding what's in each register at any given moment is essential for following program logic.

The **general-purpose registers** (RAX, RBX, RCX, RDX, RSI, RDI, R8-R15) are used for arithmetic, data manipulation, and passing function arguments. In the Windows x64 calling convention (which you'll learn more about in [Lesson 2: x86-64 Assembly Refresher](lesson_02_x86_64_refresher.md)), the first four integer arguments to a function are passed in RCX, RDX, R8, and R9. The return value is typically placed in RAX. Watching these registers as you step through function calls helps you understand what data is being passed around.

The **RSP register** (stack pointer) points to the top of the stack—the most recently pushed value. The stack grows downward in memory, so when you push a value, RSP decreases. When you pop a value, RSP increases. Monitoring RSP helps you understand stack operations and detect stack corruption.

The **RBP register** (base pointer) is traditionally used to mark the base of the current stack frame, making it easier to access local variables and function parameters at fixed offsets. However, modern compilers often omit the frame pointer for optimization, using RSP-relative addressing instead.

The **RIP register** (instruction pointer) contains the address of the current instruction—the one that's about to execute. You can't modify RIP directly with most instructions, but control flow instructions like `jmp`, `call`, and `ret` change it. Watching RIP helps you understand the program's execution path.

The **flags register** contains individual bits that indicate the results of operations. The most important flags are ZF (zero flag, set when a result is zero), CF (carry flag, set when an arithmetic operation carries or borrows), SF (sign flag, set when a result is negative), and OF (overflow flag, set when signed arithmetic overflows). These flags control conditional jumps—for example, `je` (jump if equal) checks the ZF flag. Understanding flags is crucial for understanding conditional logic.

### Memory

While registers hold a tiny amount of data, memory holds everything else—code, data, the stack, the heap. x64dbg lets you view memory at any address, which is essential for understanding what the program is doing with data.

To view memory at a specific address, right-click in the disassembly view and select "Follow in Dump", or press Ctrl+G to open the "Go to" dialog. Enter an address (either a literal hex address like `0x140001000` or an expression like `rax` to view memory at the address contained in RAX), and x64dbg will display the memory contents in the dump panel.

The memory dump shows both the raw bytes and an ASCII interpretation. This dual view is useful because sometimes you're looking for binary data (like pointers or integers), and sometimes you're looking for strings. You can also right-click in the dump panel to change the display format—for example, viewing memory as 32-bit or 64-bit integers instead of bytes.

### The Stack

The stack is a special region of memory used for function calls, local variables, and temporary storage. It's displayed in the left panel of x64dbg, showing the memory around the current stack pointer (RSP). Understanding the stack is fundamental to reverse engineering because it's where most of the action happens during function calls.

When you look at the stack panel, you'll see addresses on the left and values on the right. The topmost entry (at the lowest address, since the stack grows downward) is the current top of the stack—the value that would be popped if you executed a `pop` instruction. Below that, you'll see previously pushed values, including function arguments, saved registers, and return addresses.

Return addresses are particularly important—they're the addresses that the program will jump to when the current function returns. If you're inside a function and you look at the stack, you can often find the return address and use it to understand who called this function. x64dbg even annotates return addresses in the stack view, making them easy to identify.

Local variables are also stored on the stack, typically at negative offsets from RBP (if a frame pointer is used) or at positive offsets from RSP. By examining the stack, you can see the values of local variables as they change during function execution, which is invaluable for understanding program logic.

## Modifying Values

One of the most powerful features of a debugger is the ability to modify the program's state while it's running. You can change register values, modify memory contents, and even alter the instruction pointer to skip or repeat code. This capability is invaluable for testing hypotheses about how the program works and for bypassing checks or protections.

### Modifying Registers

To modify a register value, locate the register in the registers panel (left side of x64dbg) and right-click on it. Select "Modify value" or simply double-click the register. A dialog will appear where you can enter a new value in hexadecimal. For example, if you want to test what happens when RAX contains 0, you can set it to 0 and then continue execution.

This is particularly useful when analyzing conditional logic. Suppose you encounter a comparison like `cmp rax, 0` followed by `je skip_code`. You can set a breakpoint at the comparison, let it execute, and then manually set the zero flag (ZF) to control which branch is taken. This lets you explore both code paths without needing to manipulate the program's inputs or recompile anything.

### Modifying Memory

Memory modification works similarly. In the memory dump panel, right-click on any byte and select "Edit" or "Binary Edit" for more advanced editing. You can change individual bytes, or you can edit larger chunks of data. For example, if you find a string in memory that says "Trial Version", you could change it to "Full Version" to see if that affects the program's behavior.

Memory modification is also useful for bypassing simple protections. If a program checks a global variable to see if it's registered, you can find that variable in memory and change its value from 0 (unregistered) to 1 (registered). While this doesn't give you a permanent crack, it lets you explore the program's full functionality and understand how the protection works.

This capability to modify values at runtime is powerful for testing different code paths without recompiling. Instead of changing source code, recompiling, and running again, you can simply modify values in the debugger and immediately see the results. This rapid iteration is one of the key advantages of dynamic analysis over static analysis alone.

## Tracing Function Calls

When you need to understand the overall flow of a program rather than examining individual instructions, x64dbg's tracing feature becomes invaluable. Tracing records the execution path of the program, creating a log of every instruction (or every function call) that executes. This gives you a bird's-eye view of what the program is doing, which is especially useful for understanding complex control flow or finding where specific functionality is implemented.

To start tracing, go to Debug → Trace in the menu. You'll be presented with options for what to trace. "Trace into" records every single instruction that executes, including stepping into function calls. This creates a very detailed trace but can be overwhelming for long-running programs. "Trace over" records instructions but treats function calls as single steps, similar to how Step Over works. This creates a more manageable trace that focuses on the high-level flow.

Once tracing is active, x64dbg records all executed instructions in a trace log. You can review this log to see the exact sequence of instructions that executed, which is useful for understanding loops, recursive functions, or complex branching logic. The trace also shows register values at each step, so you can see how data flows through the program.

Tracing is particularly useful when you're trying to find where a specific action occurs. For example, if you're analyzing a program and you want to know where it writes to a file, you could set a breakpoint on the `WriteFile` API function, run the program, and then review the trace to see the call stack and execution path that led to that function call. This technique, combined with strategic breakpoints, is one of the most effective ways to understand unfamiliar code.

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

When you set a breakpoint at the start of `main` and run the program (F9), x64dbg will pause execution at the first instruction of your main function. At this point, you can inspect the complete state of the program, which provides valuable insights into how Windows sets up a program's execution environment.

**Registers**: Looking at the registers panel, you'll see various values that have been set up by the runtime initialization code that runs before `main`. The RAX and RBX registers may contain arbitrary values left over from previous operations—these general-purpose registers aren't guaranteed to have any specific value at function entry unless they're being used to pass arguments.

The RCX and RDX registers are more interesting. In the Windows x64 calling convention, these are the first two argument registers. For a typical `main` function, RCX contains `argc` (the argument count) and RDX contains `argv` (a pointer to the argument array). If you run your program without command-line arguments, RCX will typically be 1 (the program name itself counts as an argument).

The RSP register (stack pointer) points to the top of the stack. If you look at the value in the stack panel, you'll see it points to the return address—the location where execution will continue when `main` returns. This is typically somewhere in the C runtime library code that called `main`.

The RBP register (base pointer) may or may not be set up yet, depending on whether your compiler uses frame pointers. Modern compilers often omit frame pointers for optimization, using RSP-relative addressing instead. If you see RBP being set up with `push rbp; mov rbp, rsp` at the start of `main`, that's the function prologue establishing a stack frame.

The RIP register (instruction pointer) contains the address of the current instruction—the first instruction of `main`. This is the address you'll see highlighted in the disassembly view.

**Stack**: The stack panel shows the memory around RSP. At the very top (lowest address), you'll see the return address—where the program will jump when `main` executes a `ret` instruction. This address is typically in `__scrt_common_main_seh` or similar runtime initialization code. Below the return address, you might see shadow space (32 bytes reserved for the first four arguments, even though they're passed in registers) and space for local variables, though these may not be initialized yet.

As you step through the function, you'll see local variables being allocated (by subtracting from RSP) and initialized. Watching the stack grow and shrink as functions are called and return is one of the best ways to understand how the call stack works.

### Solution 2: Step Through a Function

When you step through `main` using F10 (Step Over) and F11 (Step Into), you'll observe the program executing one instruction at a time. This gives you an intimate understanding of how high-level code translates to assembly and how the CPU actually executes your program.

Each instruction you step through will be highlighted in the disassembly view. Watch the registers panel carefully—you'll see registers changing as values are computed. For example, if you see `mov rax, 5`, the RAX register will immediately update to show 5. If you see `add rax, 3`, RAX will change from 5 to 8.

Memory changes are equally important to observe. When you encounter instructions like `mov [rsp+20h], rax`, you're seeing a value being stored to the stack. If you have the stack panel visible, you'll see the memory at that location update in real-time. This is particularly interesting when watching string operations or structure initialization.

Function calls are especially educational to step through. When you encounter a `call` instruction, you have a choice: Step Into (F11) to see the function's implementation, or Step Over (F10) to execute the entire function and pause at the next instruction. If you Step Into a function, watch what happens to RSP (it decreases as the return address is pushed) and RIP (it jumps to the function's address). When the function returns, RSP increases back to its previous value, and RIP jumps to the instruction after the `call`.

Pay special attention to the calling convention in action. Before a function call, you'll see arguments being loaded into RCX, RDX, R8, and R9 (for the first four arguments). After the function returns, you'll see the return value in RAX. This pattern repeats for every function call, and recognizing it becomes second nature with practice.

### Solution 3: Modify a Value and Change Behavior

Modifying values at runtime is one of the most powerful techniques in dynamic analysis, and this exercise demonstrates how program behavior is entirely determined by the values in registers and memory at any given moment.

Let's say you've found a conditional branch in your program—perhaps an `if` statement that checks whether the user's input is empty. In assembly, this might look like:

```
cmp rax, 0          ; Compare RAX to 0
je skip_code        ; Jump if equal (if RAX is 0)
; code to execute if RAX is not 0
skip_code:
```

Set a breakpoint on the `cmp` instruction and run the program. When it pauses, look at the value in RAX. Let's say it contains 5 (meaning the input is not empty). Normally, the `je` (jump if equal) would not be taken because RAX is not equal to 0.

Now, modify RAX to be 0. Right-click on RAX in the registers panel, select "Modify value", and enter 0. Step over the `cmp` instruction (which will set the zero flag because 0 - 0 = 0), and then step over the `je` instruction. You'll see that the jump IS taken, even though the original value in RAX was 5. The program skips the code that would normally execute and jumps directly to `skip_code`.

This demonstrates a fundamental principle: the program's behavior is entirely determined by the current state of registers and memory. By modifying that state, you can make the program take different code paths, bypass checks, or test edge cases without changing the binary or recompiling anything.

This technique is invaluable for understanding protections. If you encounter a license check that compares a serial number, you can modify the comparison result to always succeed, allowing you to explore the full program functionality. While this doesn't give you a permanent crack, it helps you understand how the protection works and what the program does when it thinks it's properly licensed.

You can also use this technique to test error handling. If you want to see what happens when a function fails, you can modify its return value to indicate failure, even if it actually succeeded. This lets you explore code paths that might be difficult to trigger through normal program operation.

## Summary

You now know how to use x64dbg for dynamic analysis. You can:

- Launch binaries in the debugger
- Set and use breakpoints
- Step through code
- Inspect registers and memory
- Modify values at runtime
- Trace function calls

In the next lesson, you'll learn how to patch binaries—modifying them to change their behavior.
