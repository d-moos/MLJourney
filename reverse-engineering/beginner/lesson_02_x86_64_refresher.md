# Lesson 2: x86-64 Assembly Refresher for Reversers

## Overview

This lesson gives you a focused refresher on **x86-64 assembly** as it appears in compiler-generated code on Windows. You will not become an assembly expert, but you will learn to recognize common patterns: function prologues and epilogues, conditionals, loops, and function calls. This knowledge is essential for reading disassembly in Binary Ninja and understanding what a binary does.

The key insight is that **assembly is just a low-level representation of high-level code**. Every `if` statement, loop, and function call has a predictable assembly pattern. Once you learn these patterns, you can reverse them: see the assembly and reconstruct the original logic.

## What You'll Learn

By the end of this lesson, you will understand:

- **Registers**: What they are, which ones are important, and what they're used for
- **Flags**: How status flags (ZF, CF, SF, OF) work and how they control conditional jumps
- **The stack**: How `push`, `pop`, `call`, and `ret` work together
- **Function prologues and epilogues**: The standard code at the start and end of functions
- **The Windows x64 calling convention**: How arguments are passed to functions and how return values are passed back
- **Common patterns**: How if/else, loops, and function calls look in assembly
- **How to map assembly back to source code**: The core skill of reverse engineering

## Prerequisites

Before starting this lesson, you should:

- Be comfortable reading simple C or Rust code
- Have a basic idea of what assembly is (you don't need to be an expert)
- Have completed Lesson 1 (lab setup)

## Registers: The CPU's Scratch Pads

A register is a tiny piece of ultra-fast memory built into the CPU. x86-64 has 16 general-purpose registers, each 64 bits (8 bytes) wide. Here are the most important ones:

**RAX** (Accumulator) is the primary register for arithmetic and return values. When a function returns a value, it goes in RAX.

**RBX** (Base) is a general-purpose register. It's often used to hold base addresses or loop counters.

**RCX** (Counter) is traditionally used for loop counters, but on Windows x64, it holds the **first function argument**.

**RDX** (Data) is a general-purpose register. On Windows x64, it holds the **second function argument**.

**RSP** (Stack Pointer) points to the top of the stack. It's automatically updated by `push`, `pop`, `call`, and `ret` instructions. You rarely modify it directly.

**RBP** (Base Pointer) points to the base of the current stack frame. It's used to access local variables and function arguments.

**RSI** (Source Index) and **RDI** (Destination Index) are used for string operations and general-purpose work.

**R8–R15** are additional general-purpose registers. On Windows x64, R8 and R9 hold the **third and fourth function arguments**.

Each register can also be accessed in smaller sizes:
- **RAX** (64-bit), **EAX** (32-bit), **AX** (16-bit), **AL** (8-bit)
- Same pattern for RBX, RCX, RDX, RSI, RDI, RBP, RSP

When you write to a 32-bit register (e.g., `mov eax, 5`), the upper 32 bits of the 64-bit register are automatically zeroed. This is a common compiler optimization.

## Status Flags

Status flags are single-bit indicators that record the result of the last operation. The most important ones are:

**ZF** (Zero Flag) is set to 1 if the result of the last operation was zero, and 0 otherwise. For example, after `cmp rax, rbx` (which subtracts rbx from rax), ZF is set if they were equal.

**CF** (Carry Flag) is set if the last operation produced a carry (overflow for unsigned arithmetic).

**SF** (Sign Flag) is set if the result was negative (the most significant bit is 1).

**OF** (Overflow Flag) is set if the result overflowed (for signed arithmetic).

Conditional jumps use these flags to decide whether to jump. For example:
- `je` (jump if equal) jumps if ZF is set
- `jne` (jump if not equal) jumps if ZF is clear
- `jl` (jump if less) jumps if SF != OF
- `jg` (jump if greater) jumps if ZF is clear and SF == OF

## The Stack

The stack is a region of memory used for temporary storage. It grows downward (from high addresses to low addresses) on x86-64. The **RSP** register always points to the top of the stack (the most recently pushed value).

**push** decrements RSP by 8 and writes a value to the memory location RSP now points to. For example:
```
push rax          ; RSP -= 8; memory[RSP] = RAX
```

**pop** reads a value from the memory location RSP points to and increments RSP by 8. For example:
```
pop rax           ; RAX = memory[RSP]; RSP += 8
```

**call** pushes the return address (the address of the next instruction) onto the stack and jumps to the target function. For example:
```
call some_function  ; push (address of next instruction); jmp some_function
```

**ret** pops the return address from the stack and jumps to it. For example:
```
ret               ; pop rax; jmp rax
```

## Function Prologues and Epilogues

Every function starts with a **prologue** and ends with an **epilogue**. These are standard sequences of instructions that set up and tear down the function's stack frame.

A typical prologue looks like:
```
push rbp          ; Save the old base pointer
mov rbp, rsp      ; Set up new base pointer
sub rsp, 0x20     ; Allocate space for local variables
```

This does three things:
1. Saves the old RBP (so we can restore it when the function returns)
2. Sets RBP to point to the current stack frame
3. Allocates space for local variables by subtracting from RSP

A typical epilogue looks like:
```
mov rsp, rbp      ; Restore RSP
pop rbp           ; Restore the old base pointer
ret               ; Return to caller
```

This reverses the prologue:
1. Restores RSP
2. Restores the old RBP
3. Returns to the caller

Once you recognize these patterns, you can quickly identify function boundaries in disassembly.

## Windows x64 Calling Convention

The **calling convention** is a set of rules about how functions receive arguments and return values. On Windows x64, the rules are:

**First four integer/pointer arguments** are passed in registers:
- 1st argument: RCX
- 2nd argument: RDX
- 3rd argument: R8
- 4th argument: R9

**Additional arguments** are passed on the stack (pushed by the caller).

**Return value** is passed back in RAX (for 64-bit values) or RAX:RDX (for 128-bit values).

**Caller-saved registers** (RAX, RCX, RDX, R8–R11) can be freely modified by the function. If the caller needs to preserve them, it must save them before calling the function.

**Callee-saved registers** (RBX, RSP, RBP, RSI, RDI, R12–R15) must be preserved by the function. If the function modifies them, it must save and restore them.

**Shadow space**: The caller must allocate 32 bytes (4 * 8 bytes) of space on the stack above the return address. This is used by the function to spill arguments if needed.

## Mapping Code to Assembly

Let's look at some concrete examples of how high-level code maps to assembly.

### Example 1: Simple Function Call

**Rust code**:
```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = add(5, 3);
}
```

**Assembly** (simplified):
```
main:
    mov ecx, 5          ; 1st argument (a) = 5
    mov edx, 3          ; 2nd argument (b) = 3
    call add            ; Call add function
    ; result is now in eax
    ret

add:
    add ecx, edx        ; ecx = a + b
    mov eax, ecx        ; return value = ecx
    ret
```

Notice:
- Arguments are passed in ECX and EDX (the 32-bit versions of RCX and RDX)
- The return value is in EAX (the 32-bit version of RAX)
- `call` pushes the return address and jumps to `add`
- `ret` pops the return address and jumps back

### Example 2: If/Else Statement

**Rust code**:
```rust
fn check_positive(x: i32) -> &'static str {
    if x > 0 {
        "positive"
    } else {
        "non-positive"
    }
}
```

**Assembly** (simplified):
```
check_positive:
    cmp ecx, 0          ; Compare x with 0
    jle not_positive    ; Jump if x <= 0

    ; x > 0 branch
    lea rax, [positive_str]  ; Load address of "positive"
    ret

not_positive:
    lea rax, [non_positive_str]  ; Load address of "non-positive"
    ret
```

Notice:
- `cmp ecx, 0` compares the argument with 0 and sets flags
- `jle` (jump if less or equal) jumps if the condition is false
- Each branch loads a different string address into RAX
- Both branches return with the result in RAX

### Example 3: Loop

**Rust code**:
```rust
fn sum_to_n(n: i32) -> i32 {
    let mut sum = 0;
    for i in 0..n {
        sum += i;
    }
    sum
}
```

**Assembly** (simplified):
```
sum_to_n:
    xor eax, eax        ; sum = 0 (eax = 0)
    xor edx, edx        ; i = 0 (edx = 0)
    cmp edx, ecx        ; Compare i with n
    jge loop_end        ; Jump if i >= n

loop_start:
    add eax, edx        ; sum += i
    inc edx              ; i++
    cmp edx, ecx        ; Compare i with n
    jl loop_start       ; Jump if i < n

loop_end:
    ret
```

Notice:
- `xor eax, eax` sets eax to 0 (a common way to zero a register)
- The loop has a header (initial comparison), body (add and increment), and exit condition
- `jl` (jump if less) jumps back to the loop start if the condition is true
- When the loop exits, the result is in EAX

## Recognizing Patterns in Binary Ninja

When you open a binary in Binary Ninja, you'll see disassembly like the examples above. Here's how to recognize common patterns:

**Function prologue**: Look for `push rbp; mov rbp, rsp; sub rsp, ...`

**Function epilogue**: Look for `mov rsp, rbp; pop rbp; ret`

**If/else**: Look for `cmp` followed by a conditional jump (`je`, `jne`, `jl`, `jg`, etc.)

**Loop**: Look for a backward jump (`jl`, `jg`, `jne`, etc.) that jumps to an earlier instruction

**Function call**: Look for `call` followed by a function name or address

**Return value**: Look for a value being moved into RAX before a `ret`

## Exercises

### Exercise 1: Identify Registers in a Simple Function

**Objective**: Understand how function arguments and return values use registers.

**Steps**:
1. Create a Rust file `add.rs`:
   ```rust
   fn add(a: i32, b: i32) -> i32 {
       a + b
   }

   fn main() {
       let result = add(10, 20);
       println!("Result: {}", result);
   }
   ```

2. Compile it: `rustc -O add.rs`

3. Open `add.exe` in Binary Ninja

4. Find the `add` function (search for "add" in the functions list)

5. Identify:
   - Which register holds the first argument (a)
   - Which register holds the second argument (b)
   - Which register holds the return value
   - The prologue and epilogue

6. Document your findings in a text file

**Verification**: You should identify that:
- First argument is in ECX (or RCX)
- Second argument is in EDX (or RDX)
- Return value is in EAX (or RAX)

### Exercise 2: Reconstruct an If/Else

**Objective**: Map assembly back to high-level code.

**Steps**:
1. Create a Rust file `check.rs`:
   ```rust
   fn is_even(x: i32) -> bool {
       if x % 2 == 0 {
           true
       } else {
           false
       }
   }

   fn main() {
       println!("5 is even: {}", is_even(5));
       println!("4 is even: {}", is_even(4));
   }
   ```

2. Compile it: `rustc -O check.rs`

3. Open `check.exe` in Binary Ninja

4. Find the `is_even` function

5. Identify:
   - The `cmp` instruction that performs the comparison
   - The conditional jump that branches based on the result
   - The two branches (true and false)
   - How the return value is set in each branch

6. Write a summary of what the assembly does

**Verification**: You should understand that:
- The function compares x with 0 (or performs a modulo operation)
- A conditional jump branches to different code based on the result
- Each branch sets the return value differently

### Exercise 3: Trace a Loop

**Objective**: Identify loop structure in assembly.

**Steps**:
1. Create a Rust file `loop_sum.rs`:
   ```rust
   fn sum_range(start: i32, end: i32) -> i32 {
       let mut sum = 0;
       let mut i = start;
       while i < end {
           sum += i;
           i += 1;
       }
       sum
   }

   fn main() {
       let result = sum_range(1, 5);
       println!("Sum: {}", result);
   }
   ```

2. Compile it: `rustc -O loop_sum.rs`

3. Open `loop_sum.exe` in Binary Ninja

4. Find the `sum_range` function

5. Identify:
   - The loop header (initial setup and first comparison)
   - The loop body (the code that repeats)
   - The loop exit condition (the jump that exits the loop)
   - The backward jump that repeats the loop

6. Draw a diagram showing the loop structure

**Verification**: You should identify:
- A comparison instruction (cmp)
- A conditional jump that exits the loop
- A backward jump that repeats the loop
- The loop body between the backward jump and the exit condition

## Solutions

### Solution 1: Registers in a Simple Function

When you open `add.exe` in Binary Ninja and find the `add` function, you should see something like:

```
add:
    add ecx, edx        ; ecx = a + b
    mov eax, ecx        ; eax = result
    ret
```

Or with a prologue/epilogue:

```
add:
    push rbp
    mov rbp, rsp
    add ecx, edx        ; ecx = a + b
    mov eax, ecx        ; eax = result
    pop rbp
    ret
```

**Key observations**:
- First argument (a) is in ECX
- Second argument (b) is in EDX
- Return value is moved to EAX before returning
- The function doesn't need local variables, so the prologue might be minimal

### Solution 2: If/Else Reconstruction

The `is_even` function should show:

```
is_even:
    mov eax, ecx        ; eax = x
    cdq                 ; Sign extend eax to edx:eax
    mov ecx, 2
    idiv ecx            ; eax = x / 2, edx = x % 2
    mov eax, 0
    cmp edx, 0          ; Compare remainder with 0
    je is_even_true     ; Jump if equal (even)
    mov eax, 0          ; Return false
    ret
is_even_true:
    mov eax, 1          ; Return true
    ret
```

**Key observations**:
- The function performs a modulo operation (idiv)
- `cmp edx, 0` compares the remainder with 0
- `je` jumps if the remainder is 0 (even)
- Each branch sets EAX to 0 (false) or 1 (true)

### Solution 3: Loop Structure

The `sum_range` function should show:

```
sum_range:
    xor eax, eax        ; sum = 0
    mov r8d, ecx        ; i = start (ecx)
    cmp r8d, edx        ; Compare i with end (edx)
    jge loop_end        ; Jump if i >= end

loop_start:
    add eax, r8d        ; sum += i
    inc r8d              ; i++
    cmp r8d, edx        ; Compare i with end
    jl loop_start       ; Jump if i < end

loop_end:
    ret
```

**Key observations**:
- `xor eax, eax` initializes sum to 0
- The first `cmp` and `jge` check if the loop should run at all
- The loop body adds i to sum and increments i
- The second `cmp` and `jl` check if the loop should continue
- The backward jump (`jl loop_start`) repeats the loop

## Summary

You now understand the basics of x86-64 assembly as it appears in Windows binaries. You know:

- How registers work and which ones are important
- How status flags control conditional jumps
- How the stack works and how functions use it
- The Windows x64 calling convention
- How common high-level constructs map to assembly
- How to recognize patterns in disassembly

In the next lesson, you'll learn about the PE file format—the structure that wraps this assembly code into a Windows executable.

