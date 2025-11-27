# Lesson 3: VM Unpacking – Extracting Code from Virtual Machines

## Overview

**VM unpacking** is the process of extracting code from VM-protected binaries. This is more complex than simple unpacking because:
- The code is virtualized (compiled to bytecode)
- The VM is custom and proprietary
- Anti-debugging and anti-VM techniques are present
- The bytecode must be decompiled

## What You'll Learn

By the end of this lesson, you will understand:

- **How to identify VM entry points**
- **How to dump VM bytecode**
- **How to reverse engineer the VM instruction set**
- **How to decompile bytecode**
- **How to handle anti-debugging during unpacking**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-2 of this course
- Understand VM-based virtualization
- Be comfortable with advanced debugging

## Identifying VM Entry Points

VM entry points are the locations in the code where execution transitions from native x86-64 instructions to virtualized bytecode. Finding these entry points is the first critical step in VM unpacking because they mark the boundaries between code you can analyze normally and code that's been transformed into a custom instruction set.

### Characteristics of VM Entry Points

VM entry points have distinctive characteristics that set them apart from normal function calls. **Unusual code patterns** are often the first clue—instead of the typical function prologue (`push rbp; mov rbp, rsp`), you might see complex register initialization, unusual stack manipulations, or immediate jumps to computed addresses. The code might set up multiple registers with specific values that serve as the VM's initial state, such as a bytecode instruction pointer, a virtual stack pointer, and flags.

**Large data sections nearby** is another strong indicator. VM-protected code stores the bytecode somewhere in memory, typically in a data section or a specially allocated region. If you see a function that references a large array or buffer of data, and that data has high entropy (indicating it might be encrypted bytecode), you're likely looking at a VM entry point. The function will read from this data section as it interprets the bytecode.

**Complex register setup** is characteristic of VM entry points because the VM needs to initialize its virtual machine state. You might see a sequence of instructions that loads specific values into multiple registers—for example, RBX might hold the bytecode pointer, RSI might hold the virtual stack pointer, and RDI might hold a pointer to a context structure. This initialization is more complex than a normal function call, which typically only sets up a few argument registers.

**Calls to VM interpreter functions** are the most definitive sign. The VM interpreter is the core engine that reads bytecode and executes the corresponding operations. If you see a call to a large, complex function that's called repeatedly from multiple locations, and that function contains a large switch statement or jump table, you've likely found the VM interpreter. The calls to this interpreter are the VM entry points.

### Finding VM Entry Points

Finding VM entry points requires a combination of static and dynamic analysis. Start by **looking for suspicious functions** in your static analysis tool. In Binary Ninja, scan through the function list looking for functions with unusual characteristics—very large functions (the VM interpreter itself), functions with many basic blocks (indicating complex control flow), or functions that reference large data arrays.

**Analyze the entry point** of the protected binary. Many VM protectors virtualize the main entry point or critical functions like license checks. Load the binary in Binary Ninja and navigate to the entry point. If the code immediately looks unusual—complex register setup, references to large data sections, or calls to suspicious functions—the entry point itself might be virtualized. Trace the control flow from the entry point to see where it leads.

**Look for the VM interpreter** by identifying large functions with characteristic patterns. The VM interpreter typically contains a large switch statement or jump table that dispatches to different handlers based on the current bytecode instruction. In disassembly, this appears as a comparison of a value (the opcode) followed by a series of conditional jumps or a computed jump through a table. Functions with this pattern, especially if they're called from many locations, are likely VM interpreters.

**Trace execution dynamically** using x64dbg to find VM entry points at runtime. Set a breakpoint at the program's entry point and step through execution. Watch for the transition from normal-looking code to the unusual patterns described above. When you see complex register setup followed by a call to a large function, you've likely found a VM entry point. Note the address and the function being called—you'll need these for further analysis.

You can also use tracing to find VM entry points. Enable instruction tracing in x64dbg and let the program run for a while, then examine the trace log. Look for patterns where execution repeatedly enters and exits the same large function—this is likely the VM interpreter being called for each bytecode instruction. The locations that call this interpreter are your VM entry points.

## Dumping VM Bytecode

Once you've identified a VM entry point and located the VM interpreter, the next step is to dump the bytecode from memory. The bytecode is the virtualized version of the original code, and analyzing it is essential for understanding what the protected function actually does.

**Set a breakpoint** at the VM entry point you identified. This is typically the instruction just before the call to the VM interpreter, or the first instruction of the VM interpreter itself. The goal is to pause execution at a point where the bytecode is fully decrypted and loaded in memory, but before the VM starts executing it.

**Run the program** (F9 in x64dbg) until it hits your breakpoint. At this point, the VM is initialized and ready to execute bytecode. Now you need to identify where the bytecode is stored in memory. Look at the registers—one of them (often RBX, RSI, or a similar register) will contain a pointer to the bytecode. You can identify this by looking at what the VM interpreter reads from. Step through a few instructions of the VM interpreter and watch which memory addresses it accesses. The address it reads from repeatedly is likely the bytecode pointer.

**Identify the bytecode location** by examining the memory at the address you found. In x64dbg, right-click in the disassembly view and select "Follow in Dump" to view the memory at that address. You should see data that looks like bytecode—sequences of bytes that don't correspond to valid x86-64 instructions. The bytecode might be encrypted, but if you're at the right point in execution, it should be decrypted.

**Dump the bytecode** to a file for offline analysis. In x64dbg, right-click in the memory dump view and select "Binary" → "Save to file". You'll need to determine the size of the bytecode—this might require some experimentation. Look for patterns that indicate the end of the bytecode, such as a terminator byte, a return instruction, or a transition back to native code. Start with a conservative estimate (like 1 KB or 4 KB) and adjust as needed.

**Analyze the bytecode** to understand its structure. Open the dumped file in a hex editor and look for patterns. Are there repeating byte sequences? Do certain bytes appear more frequently than others (these might be common opcodes)? Are there obvious data embedded in the bytecode (like string references or constants)? This initial analysis helps you understand the bytecode format before you start reverse engineering the instruction set.

## Reverse Engineering the VM Instruction Set

Understanding the VM instruction set is the most challenging part of VM unpacking. Each VM protector uses a custom instruction set, so you can't rely on existing documentation—you have to reverse engineer it from scratch by observing how the VM interpreter behaves.

**Identify the instruction format** by examining how the VM interpreter reads bytecode. Set a breakpoint at the start of the VM interpreter's main loop and step through it carefully. Watch how it reads from the bytecode pointer. Does it read one byte at a time, or multiple bytes? Does it read a fixed-size instruction, or does the instruction size vary based on the opcode? For example, you might see:

```
mov al, [rbx]      ; Read opcode (1 byte)
inc rbx            ; Advance bytecode pointer
```

This indicates a variable-length instruction format where the opcode is 1 byte, and operands (if any) follow. Alternatively, you might see the VM read 4 or 8 bytes at once, indicating fixed-size instructions.

**Identify instruction types** by analyzing the VM interpreter's dispatch mechanism. Most VM interpreters use a switch statement or jump table to dispatch to different handlers based on the opcode. Find this dispatch mechanism—it's usually a comparison followed by conditional jumps, or a computed jump through a table. Each case in the switch (or each entry in the jump table) corresponds to one instruction type. Count how many cases there are—this tells you how many different instruction types the VM supports.

**Identify operands** by examining individual instruction handlers. Pick a simple-looking handler and analyze what it does. Does it read additional bytes from the bytecode after the opcode? These are operands. Does it use immediate values, or does it reference registers or memory? For example, a handler might read two more bytes after the opcode, interpret them as register indices, and perform an operation on those virtual registers. Document the operand format for each instruction type you analyze.

**Write a disassembler** to automate the process of converting bytecode to a readable format. This can be a simple Python script that reads the bytecode file, parses each instruction according to the format you've identified, and prints it in a human-readable format. For example:

```
0x0000: PUSH 0x1234
0x0003: PUSH 0x5678
0x0006: ADD
0x0007: POP [0x10]
```

Your disassembler doesn't need to be perfect initially—start with the instructions you understand and gradually add support for more as you reverse engineer additional handlers.

**Analyze patterns** in the disassembled bytecode to understand what the original code was doing. Look for common patterns like function prologues (stack frame setup), loops (backward jumps with counters), and function calls (push return address, jump to new location). These patterns help you understand the high-level structure of the code, even if you don't understand every individual instruction.

## Decompiling Bytecode

Once you have a working disassembler and understand the instruction set, you can begin the process of decompiling the bytecode back to something resembling the original source code. This is more art than science, requiring pattern recognition and understanding of common programming constructs.

**Disassemble the bytecode** using the disassembler you created. Generate a complete listing of all instructions in a readable format. This gives you a foundation to work from—instead of staring at raw bytes, you're looking at mnemonics and operands that make sense.

**Analyze control flow** by identifying jumps, branches, and function calls in the disassembled bytecode. Mark all jump targets and create a control flow graph showing how execution can flow through the bytecode. Identify basic blocks (sequences of instructions with no branches), and connect them based on the jumps. This reveals the structure of the code—loops, conditionals, and function boundaries.

Look for patterns that indicate specific constructs. A backward jump with a counter register suggests a loop. A conditional jump based on a comparison suggests an if statement. A push of an address followed by a jump suggests a function call. By identifying these patterns, you can start to reconstruct the high-level logic.

**Reconstruct pseudocode** by translating the bytecode instructions into higher-level operations. Instead of "PUSH 5; PUSH 3; ADD; POP result", write "result = 5 + 3". Instead of a sequence of bytecode instructions that implement a loop, write "for (i = 0; i < 10; i++)". This process is manual and requires understanding both the bytecode and common programming patterns.

Start with small, isolated sections of code—perhaps a single basic block or a simple function. Translate it to pseudocode, verify your understanding by comparing it to the bytecode, and then move on to more complex sections. Gradually, you'll build up a complete pseudocode representation of the virtualized function.

**Optimize and simplify** the pseudocode to make it more readable. VM-generated code often includes redundant operations, temporary variables, and convoluted control flow that wouldn't appear in hand-written code. Simplify these by eliminating dead code, combining operations, and restructuring control flow to use more natural constructs like while loops instead of goto statements.

The end result should be pseudocode that clearly expresses what the original function did, even if it doesn't exactly match the original source code. This is often sufficient for understanding the functionality, finding vulnerabilities, or bypassing protections.

## Exercises

### Exercise 1: Identify VM Entry Points

**Objective**: Learn to find VM entry points.

**Steps**:
1. Take a VM-protected binary
2. Open it in Binary Ninja
3. Look for suspicious functions
4. Identify potential VM entry points
5. Document your findings

**Verification**: You should be able to identify VM entry points.

### Exercise 2: Dump VM Bytecode

**Objective**: Learn to dump bytecode.

**Steps**:
1. Open a VM-protected binary in x64dbg
2. Find a VM entry point
3. Set a breakpoint at the entry point
4. Run the program
5. Dump the bytecode from memory
6. Save it to a file

**Verification**: You should have a file containing VM bytecode.

### Exercise 3: Reverse Engineer the VM

**Objective**: Learn to reverse engineer the VM.

**Steps**:
1. Analyze the VM bytecode
2. Identify the instruction format
3. Identify instruction types
4. Write a simple disassembler
5. Disassemble some bytecode

**Verification**: You should be able to disassemble bytecode.

## Solutions

### Solution 1: Identify VM Entry Points

Identifying VM entry points in a VM-protected binary requires careful analysis of both the code structure and execution behavior. Here's what you should observe when analyzing a typical VM-protected binary, such as one protected with VMProtect or a similar tool.

**Unusual Code Patterns**: When you open the binary in Binary Ninja and navigate to the entry point or protected functions, you'll immediately notice that the code doesn't look like normal compiled code. Instead of the typical function prologue (`push rbp; mov rbp, rsp; sub rsp, XX`), you might see complex sequences like:

```
mov rbx, offset bytecode_array
mov rsi, offset vm_context
mov edi, 0x12345678
call vm_interpreter
```

This initialization sequence is setting up the VM's state—RBX points to the bytecode, RSI points to a context structure, and EDI contains initial flags or state. The call to `vm_interpreter` is the actual VM entry point.

**Called from Main Entry Point**: VM protectors often virtualize the main entry point or critical functions. If you trace execution from the program's entry point, you'll see it quickly call into suspicious functions. In Binary Ninja, follow the control flow from the entry point—within a few function calls, you should reach code with the unusual patterns described above. These are your VM entry points.

**Large Data Sections Nearby**: VM entry points reference bytecode stored in memory. In Binary Ninja, look at the cross-references for suspicious functions. If a function references a large array or buffer (you can see this in the data view), and that buffer has high entropy, it's likely bytecode. The function that references this buffer is probably a VM entry point. You can verify the entropy using Binary Ninja's entropy analysis features—bytecode often has entropy around 6.0-7.5, higher than normal code but lower than encrypted data.

**Call VM Interpreter Functions**: The most definitive sign is calls to the VM interpreter. In your function list, look for very large functions (thousands of instructions) with complex control flow. These are likely VM interpreters. Then, search for all cross-references to these functions—every location that calls the interpreter is a VM entry point. In Binary Ninja, right-click on the interpreter function and select "Show References" to see all call sites.

For a concrete example, if you're analyzing a VMProtect-protected binary, you might find a function at address 0x140001000 that's 5000 instructions long and contains a massive switch statement. This is the VM interpreter. Searching for references to this function reveals 15 different call sites—these are your 15 VM entry points, each protecting a different function in the original program.

### Solution 2: Dump VM Bytecode

Dumping VM bytecode requires precise timing and careful observation of the VM's state. Here's a detailed walkthrough of the process using x64dbg.

**Finding the VM Entry Point**: First, you need to locate a VM entry point using the techniques from Solution 1. Let's say you've identified that the function at 0x140002000 is a VM entry point that calls the interpreter at 0x140001000. Set a breakpoint at 0x140002000 and run the program (F9).

**Identifying the Bytecode Pointer**: When the breakpoint hits, you're at the VM entry point. Step through the initialization code (F8) and watch the registers carefully. You'll see registers being loaded with specific values. For example:

```
mov rbx, 0x140050000    ; RBX now points to bytecode
mov rsi, 0x140060000    ; RSI points to VM context
call 0x140001000        ; Call VM interpreter
```

The value loaded into RBX (0x140050000 in this example) is your bytecode pointer. Note this address.

**Examining the Bytecode**: Right-click in the disassembly view and select "Follow in Dump" → "Address" and enter the bytecode address (0x140050000). The dump view will show the memory at that location. You should see data that doesn't correspond to valid x86-64 instructions—this is the bytecode. It might look like:

```
01 23 45 67 89 AB CD EF 02 11 22 33 44 55 66 77 ...
```

**Determining Bytecode Size**: Figuring out how much bytecode to dump can be tricky. One approach is to step through the VM interpreter and watch the bytecode pointer (RBX in our example). Each time the interpreter executes an instruction, it advances the bytecode pointer. Continue stepping until you see the bytecode pointer reach a certain value, then return to native code. The difference between the starting and ending bytecode pointer values is the bytecode size.

Alternatively, you can make an educated guess based on the size of the original function. If the original function was 200 bytes of x86-64 code, the bytecode might be 500-1000 bytes (VM bytecode is typically larger than native code due to the overhead of virtualization).

**Dumping to File**: Once you know the address and approximate size, dump the bytecode to a file. In x64dbg, right-click in the dump view, select "Binary" → "Save to file", and specify the size (e.g., 1024 bytes). Save it as `bytecode.bin`. You now have the bytecode extracted for offline analysis.

**Verification**: To verify you dumped the correct data, you can set a breakpoint in the VM interpreter and watch it read from the bytecode. The bytes it reads should match the bytes in your dumped file. If they don't match, you may have dumped from the wrong address or at the wrong time (perhaps before decryption).

### Solution 3: Reverse Engineer the VM

Reverse engineering a VM instruction set is a complex, iterative process. Here's a detailed example of how you might approach it, using a hypothetical simple VM as an example.

**Analyzing Bytecode Patterns**: Open your dumped bytecode file in a hex editor. Look for patterns and structure. For example, you might see:

```
01 05 00 00 00 0A 00 00 00
01 05 00 00 00 14 00 00 00
02
03 10 00 00 00
```

Notice that `01` appears twice, followed by similar patterns. This suggests `01` is an opcode (perhaps PUSH), and the following bytes are operands. The `02` and `03` appear alone, suggesting they're opcodes with no operands (perhaps ADD and POP).

**Identifying Instruction Format**: To confirm the format, analyze the VM interpreter in x64dbg. Set a breakpoint at the start of the interpreter's main loop and step through it. You might see:

```
mov al, [rbx]           ; Read opcode from bytecode
inc rbx                 ; Advance bytecode pointer
cmp al, 1               ; Compare opcode to 1
je handle_push          ; If opcode is 1, handle PUSH
cmp al, 2
je handle_add
...
```

This confirms that opcodes are 1 byte, and the interpreter uses a series of comparisons to dispatch to handlers. Now examine the `handle_push` handler:

```
handle_push:
    mov eax, [rbx]      ; Read 4-byte operand
    add rbx, 4          ; Advance bytecode pointer by 4
    push rax            ; Push operand onto virtual stack
    jmp interpreter_loop
```

This tells you that the PUSH instruction (opcode 0x01) has a 4-byte operand, and the total instruction size is 5 bytes (1 byte opcode + 4 bytes operand).

**Identifying Instruction Types**: Continue this process for each handler. You might discover:
- Opcode 0x01: PUSH <4-byte immediate> (pushes a value onto the virtual stack)
- Opcode 0x02: ADD (pops two values, adds them, pushes result)
- Opcode 0x03: POP <4-byte address> (pops a value and stores it to memory)
- Opcode 0x04: JMP <4-byte offset> (unconditional jump)
- Opcode 0x05: JZ <4-byte offset> (jump if zero flag is set)

Document each instruction type with its opcode, operands, and behavior.

**Writing a Disassembler**: With this information, you can write a simple Python disassembler:

```python
def disassemble(bytecode):
    pc = 0
    while pc < len(bytecode):
        opcode = bytecode[pc]
        pc += 1

        if opcode == 0x01:  # PUSH
            value = int.from_bytes(bytecode[pc:pc+4], 'little')
            print(f"PUSH 0x{value:08X}")
            pc += 4
        elif opcode == 0x02:  # ADD
            print("ADD")
        elif opcode == 0x03:  # POP
            addr = int.from_bytes(bytecode[pc:pc+4], 'little')
            print(f"POP [0x{addr:08X}]")
            pc += 4
        # ... more opcodes ...
```

**Disassembling Bytecode**: Run your disassembler on the dumped bytecode:

```
PUSH 0x0000000A
PUSH 0x00000014
ADD
POP [0x00000010]
```

This disassembly reveals that the virtualized code pushes 10 and 20, adds them, and stores the result (30) to memory address 0x10. In pseudocode, this is `memory[0x10] = 10 + 20`.

By continuing this process—analyzing more handlers, refining your disassembler, and disassembling more bytecode—you gradually build a complete understanding of the VM instruction set and can reconstruct the original program logic. This is the essence of VM unpacking, and while it's time-consuming, it's one of the most valuable skills in advanced reverse engineering.

## Summary

You now understand VM unpacking. You can:

- Identify VM entry points
- Dump VM bytecode
- Reverse engineer VM instruction sets
- Decompile bytecode
- Analyze VM-protected binaries

In the next lesson, you'll learn about deobfuscation strategies.
