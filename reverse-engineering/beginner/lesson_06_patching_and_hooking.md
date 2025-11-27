# Lesson 6: Patching and Hooking – Modifying Binary Behavior

## Overview

**Patching** means modifying a binary to change its behavior. This is a fundamental reverse engineering skill. You might patch a binary to:
- Skip a license check
- Change a condition to always be true
- Redirect a function call to your own code
- Modify data values

In this lesson, you'll learn several patching techniques: instruction modification, inline hooks, and IAT hooking.

## What You'll Learn

By the end of this lesson, you will understand:

- **How to modify instructions** in a binary
- **How to use NOPs** to disable code
- **How to flip conditional jumps**
- **How to create inline hooks** (overwrite function prologues)
- **How to hook the IAT** (Import Address Table)
- **How to test patches** and verify they work

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-5
- Understand x86-64 assembly
- Be comfortable with x64dbg

## Instruction Modification

The simplest patch is to modify an instruction. For example:
- Change `jne` (jump if not equal) to `je` (jump if equal)
- Change `mov rax, 0` to `mov rax, 1`
- Replace an instruction with `nop` (no operation)

### Example: Flipping a Conditional Jump

Suppose you have:
```
cmp rax, 0
jne skip_code    ; Jump if not equal
; code to execute if rax == 0
skip_code:
```

To always execute the code, you can:
1. Change `jne` to `je` (flip the condition)
2. Or replace `jne` with `nop` (remove the jump)

### Using a Hex Editor

1. Open the binary in a hex editor (e.g., HxD)
2. Find the instruction you want to modify
3. Modify the bytes
4. Save the file

For example, `jne` is encoded as `75 XX` (where XX is the jump offset). To change it to `je`, change it to `74 XX`.

### Using x64dbg

1. Set a breakpoint at the instruction
2. Run the program
3. Right-click on the instruction
4. Select "Assemble"
5. Enter the new instruction
6. x64dbg modifies the instruction in memory

## NOPs and Code Disabling

A **NOP** (no operation) instruction does nothing. It's useful for disabling code:

1. Find the instruction you want to disable
2. Replace it with `nop` (or multiple `nop`s if the instruction is longer)
3. The code is effectively skipped

For example, to disable a function call:
```
call some_function    ; Original
nop                   ; Replaced (if the call is 1 byte, use multiple nops)
```

## Inline Hooks

An **inline hook** overwrites the beginning of a function to jump to your own code. This is more complex but very powerful.

### How Inline Hooks Work

1. You write your own function (the "hook function")
2. You overwrite the first few bytes of the target function with a `jmp` to your hook function
3. Your hook function can:
   - Modify arguments
   - Call the original function
   - Modify the return value
   - Skip the original function entirely

### Example

Suppose you want to hook `GetStdHandle`:

```
Original GetStdHandle:
    push rbp
    mov rbp, rsp
    ; ... rest of function

After hooking:
    jmp my_hook_function
    ; ... rest of function (unreachable)

my_hook_function:
    ; Your code here
    ; Call original GetStdHandle if needed
    ; Return
```

## IAT Hooking

The **Import Address Table (IAT)** contains addresses of imported functions. You can modify the IAT to redirect function calls.

### How IAT Hooking Works

1. Find the IAT entry for the function you want to hook
2. Replace the address with the address of your hook function
3. When the binary calls the function, it actually calls your hook

### Example

Suppose the IAT contains:
```
IAT[0] = 0x7FFF1234  ; Address of GetStdHandle
```

To hook it:
```
IAT[0] = 0x400000    ; Address of my_hook_function
```

Now when the binary calls `GetStdHandle`, it actually calls `my_hook_function`.

## Exercises

### Exercise 1: Modify an Instruction

**Objective**: Learn to patch binaries using instruction modification.

**Steps**:
1. Compile a simple Rust program with a conditional:
   ```rust
   fn main() {
       let x = 5;
       if x > 10 {
           println!("x is greater than 10");
       } else {
           println!("x is not greater than 10");
       }
   }
   ```

2. Open the binary in Binary Ninja and find the conditional jump
3. Open the binary in a hex editor
4. Find the `jle` (jump if less or equal) instruction
5. Change it to `jmp` (unconditional jump)
6. Save the binary
7. Run the modified binary and verify the behavior changed

**Verification**: The modified binary should always print "x is greater than 10".

### Exercise 2: Disable Code with NOPs

**Objective**: Learn to disable code using NOPs.

**Steps**:
1. Compile a Rust program that calls a function:
   ```rust
   fn print_secret() {
       println!("Secret message!");
   }
   
   fn main() {
       print_secret();
       println!("Normal message");
   }
   ```

2. Open the binary in Binary Ninja and find the call to `print_secret`
3. Open the binary in a hex editor
4. Find the `call print_secret` instruction
5. Replace it with `nop` instructions
6. Save the binary
7. Run the modified binary and verify the secret message is not printed

**Verification**: The modified binary should not print "Secret message!".

### Exercise 3: Create a Simple Inline Hook

**Objective**: Learn to create inline hooks.

**Steps**:
1. Write a simple Rust program with a function you want to hook
2. Compile it
3. In x64dbg:
   - Set a breakpoint at the function
   - Modify the first instruction to `jmp` to a different location
   - Create your hook code at that location
   - Test the hook

**Verification**: Your hook should be called instead of the original function.

## Solutions

### Solution 1: Modify an Instruction

When you modify the `jle` instruction to `jmp`, the conditional jump becomes unconditional. The program will always take the "true" branch, printing "x is greater than 10" even though x is 5.

### Solution 2: Disable Code with NOPs

When you replace the `call print_secret` with `nop`, the function is never called. The program will only print "Normal message".

### Solution 3: Create a Simple Inline Hook

A simple inline hook might look like:
1. Overwrite the first instruction of the target function with `jmp my_hook`
2. In `my_hook`, do your custom logic
3. Optionally call the original function
4. Return

## Summary

You now know how to patch binaries. You can:

- Modify instructions
- Use NOPs to disable code
- Flip conditional jumps
- Create inline hooks
- Hook the IAT
- Test and verify patches

In the next lesson, you'll learn about DLLs and how to hook game functions.
