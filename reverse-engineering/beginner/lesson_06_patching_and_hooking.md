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

The simplest form of binary patching is modifying individual instructions to change program behavior. This technique is fundamental to reverse engineering because it allows you to test hypotheses about how code works and bypass protections without needing to recompile anything. Common modifications include changing conditional jumps to alter control flow (for example, changing `jne` to `je` to flip a condition), modifying immediate values to change constants (like changing `mov rax, 0` to `mov rax, 1` to alter a return value), or replacing instructions with `nop` (no operation) to effectively disable them.

### Example: Flipping a Conditional Jump

Consider a typical license check scenario where you encounter the following assembly code:

```
cmp rax, 0
jne skip_code    ; Jump if not equal
; code to execute if rax == 0
skip_code:
```

This code compares RAX to 0 and jumps over some code if RAX is not equal to 0. The code between the `jne` and `skip_code` label only executes when RAX equals 0. Perhaps this is a license check where RAX contains the result of validating a serial number—0 means valid, non-zero means invalid.

To always execute the protected code regardless of the value in RAX, you have two options. First, you can flip the condition by changing `jne` (jump if not equal) to `je` (jump if equal). This inverts the logic—now the code jumps when RAX equals 0 and falls through when RAX is non-zero, which is the opposite of the original behavior. Second, you can simply remove the jump entirely by replacing the `jne` instruction with `nop` instructions. This causes execution to always fall through to the protected code, regardless of the comparison result.

### Using a Hex Editor

Patching with a hex editor gives you direct control over the binary file on disk. This creates a permanent modification that persists across program runs, unlike memory patches which only last for the current debugging session.

Start by opening the binary in a hex editor like HxD (on Windows) or any other hex editor you prefer. You need to locate the instruction you want to modify. If you know the file offset from your static analysis tool (Binary Ninja can show you file offsets), navigate directly to that offset. Otherwise, you can search for the byte pattern of the instruction.

Once you've found the instruction, modify the bytes to change its behavior. For example, the `jne` instruction is encoded as `75 XX`, where `75` is the opcode for `jne` and `XX` is the relative jump offset (how many bytes forward to jump). To change this to `je` (jump if equal), simply change the opcode byte from `75` to `74`, leaving the offset unchanged. The instruction `je` is encoded as `74 XX`, so this single-byte change flips the conditional logic.

After making your modifications, save the file. The binary is now permanently patched. When you run it, the modified instruction will execute instead of the original. This is useful for creating cracked versions of software or bypassing protections, though you should always work in your isolated VM environment and only on software you have permission to modify.

### Using x64dbg

x64dbg provides a more interactive way to patch binaries, allowing you to modify instructions while the program is running and immediately see the effects. This is excellent for experimentation and testing different patches before committing to a permanent file modification.

Begin by setting a breakpoint at the instruction you want to modify. Run the program (F9) until it hits the breakpoint. At this point, execution is paused at the instruction you want to change. Right-click on the instruction in the disassembly view and select "Assemble" from the context menu.

A dialog box appears allowing you to enter a new instruction. Type the instruction you want (for example, `je skip_code` instead of `jne skip_code`, or `nop` to disable the instruction entirely). x64dbg will assemble your instruction and modify the bytes in memory. The change takes effect immediately—when you continue execution, the modified instruction will execute instead of the original.

This memory-based patching is temporary and only lasts for the current debugging session. When you close x64dbg or restart the program, the original instruction is back. This is perfect for testing different patches to see which one achieves the desired effect. Once you've found a patch that works, you can make it permanent by using x64dbg's "Patch file" feature, which writes the memory modifications back to the binary file on disk.

## NOPs and Code Disabling

A **NOP** (no operation) instruction is an instruction that does absolutely nothing—it simply advances the instruction pointer to the next instruction without modifying any registers, memory, or flags. Despite its apparent uselessness, NOP is incredibly valuable for binary patching because it allows you to disable code without changing the size of the binary or breaking relative jump offsets.

The process of disabling code with NOPs is straightforward. First, identify the instruction or sequence of instructions you want to disable. This might be a function call you want to skip, a check you want to bypass, or any other code you want to prevent from executing. Next, replace the instruction with one or more `nop` instructions. The x86-64 `nop` instruction is encoded as a single byte (`0x90`), so if you're replacing a multi-byte instruction, you'll need multiple NOPs to fill the space.

For example, suppose you want to disable a function call that performs a license check. The `call` instruction is typically 5 bytes long (1 byte opcode + 4 bytes for the relative address). To disable it, you would replace all 5 bytes with NOP instructions (`90 90 90 90 90`). When execution reaches this location, it will execute five NOPs in sequence, effectively doing nothing, and then continue to the next instruction. The license check function is never called, and the program continues as if the check passed.

This technique is particularly useful when you want to disable code without understanding exactly what it does. If you encounter a suspicious function call or a complex sequence of instructions that you suspect is performing a check or protection, you can simply NOP it out and see what happens. If the program works better (or at all) with the code disabled, you've found something important.

## Inline Hooks

An **inline hook** is a more sophisticated patching technique that redirects execution from a target function to your own custom code. Unlike simple instruction modification, which changes behavior by altering individual instructions, inline hooking allows you to intercept function calls, inspect or modify arguments, execute custom logic, and optionally call the original function. This is the foundation of many game hacks, anti-cheat bypasses, and instrumentation tools.

### How Inline Hooks Work

The core concept of inline hooking is to overwrite the beginning of a target function with a jump instruction that redirects execution to your hook function. Here's how the process works in detail.

First, you write your own function—the hook function—that will be executed instead of (or in addition to) the original function. This hook function can do whatever you want: log function calls, modify arguments before passing them to the original function, change return values, or completely replace the original function's behavior.

Next, you overwrite the first few bytes of the target function with a `jmp` instruction that jumps to your hook function. On x86-64, a relative jump instruction is 5 bytes (`E9 XX XX XX XX`, where the X's are the relative offset to the destination). This means you're overwriting the first 5 bytes of the original function. You need to save these original bytes somewhere because you'll need them if you want to call the original function from your hook.

When the program tries to call the target function, execution immediately jumps to your hook function instead. Your hook function now has complete control. It can modify the arguments (which are in registers RCX, RDX, R8, R9 according to the Windows x64 calling convention), perform logging or other side effects, and then decide what to do next.

If you want to call the original function, you need to execute the original bytes you saved (the ones you overwrote with the jump), and then jump to the rest of the original function (skipping the overwritten bytes). This is called a "trampoline"—a small piece of code that executes the original instructions and then jumps back to the original function.

Alternatively, your hook function can skip the original function entirely and return its own value. This is useful for bypassing checks—for example, if you hook a license validation function, your hook can simply return "true" without ever calling the original validation logic.

Finally, your hook function can modify the return value after calling the original function. This allows the original function to execute normally, but you intercept the result and change it before it's returned to the caller. For instance, you might hook a function that returns the number of remaining trial days and always return 30, giving yourself unlimited trial time.

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

The **Import Address Table (IAT)** is a data structure in PE files that contains pointers to imported functions from DLLs. When a program calls an imported function like `GetStdHandle` or `CreateFileW`, it doesn't call the function directly—instead, it calls through the IAT, which contains the actual address of the function in memory. This indirection makes IAT hooking possible and, in many ways, simpler than inline hooking.

### How IAT Hooking Works

IAT hooking works by modifying the function pointers in the Import Address Table to point to your hook function instead of the original function. This technique is elegant because it doesn't require modifying any code—you're only changing data (the function pointers in the IAT).

The process begins by finding the IAT entry for the function you want to hook. You can locate the IAT using PE analysis tools like PE-bear or programmatically by parsing the PE headers. The IAT is typically in the `.rdata` section and contains an array of 8-byte pointers (on x64) to imported functions. Each entry corresponds to one imported function, and the entries are organized by DLL.

Once you've found the IAT entry for your target function, you replace the address it contains with the address of your hook function. For example, if the IAT entry for `GetStdHandle` contains `0x7FFF12345678` (the address of the real `GetStdHandle` in `kernel32.dll`), you change it to `0x0000000140001000` (the address of your hook function in the binary's memory space).

When the program calls the hooked function, it reads the address from the IAT and jumps to it—but now that address points to your hook function instead of the original. Your hook executes, and you have complete control. You can log the call, modify arguments, call the original function (using the original address you saved), modify the return value, or skip the original function entirely.

### Example

Consider a practical example where you want to hook `GetStdHandle` to log every time the program accesses standard input/output handles. The IAT initially contains:

```
IAT[0] = 0x7FFF12345678  ; Address of GetStdHandle in kernel32.dll
```

You write a hook function at address `0x140001000` that logs the call and then calls the original `GetStdHandle`. To install the hook, you modify the IAT:

```
IAT[0] = 0x140001000    ; Address of my_hook_function
```

Now, when the program executes code like `call [IAT+0]` (which is how imported functions are called), it jumps to `0x140001000` instead of `0x7FFF12345678`. Your hook function executes, logs the call, calls the original `GetStdHandle` at `0x7FFF12345678`, and returns the result. From the program's perspective, everything works normally, but you've intercepted the call.

IAT hooking is particularly useful because it's relatively easy to implement, doesn't require complex trampolines like inline hooking, and works well for hooking imported functions. However, it only works for imported functions (not internal functions), and sophisticated programs can detect IAT modifications by comparing the IAT to the original values.

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

When you successfully complete this exercise, you'll have transformed a conditional branch into an unconditional one, demonstrating how a single-byte change can completely alter program behavior.

In Binary Ninja, when you navigate to the `main` function, you'll see the conditional logic for the if statement. The assembly will include a comparison instruction like `cmp eax, 10` (comparing x to 10) followed by a conditional jump like `jle else_branch` (jump if less than or equal). This jump is taken when x is less than or equal to 10, which causes the program to skip the "x is greater than 10" message and jump to the else branch.

The `jle` instruction is encoded as `7E XX` in machine code, where `7E` is the opcode for "jump if less or equal" and `XX` is the relative offset to the jump target. To make this jump unconditional, you need to change it to `jmp`, which is encoded as `EB XX`. Notice that only the opcode byte changes—the offset remains the same because you're jumping to the same location, just unconditionally instead of conditionally.

Open the binary in a hex editor and navigate to the file offset of the `jle` instruction (Binary Ninja can show you the file offset). Find the byte `7E` and change it to `EB`. Save the file and run the modified binary. You'll see that it now prints "x is greater than 10" even though x is 5. The comparison still happens (`cmp eax, 10` still executes), but the jump is now unconditional, so the program always takes the "true" branch regardless of the comparison result.

This demonstrates a fundamental principle of binary patching: you don't need to understand or modify the entire program—a surgical change to a single byte can achieve your goal. This technique is commonly used to bypass license checks, disable trial limitations, or alter program behavior for testing purposes.

### Solution 2: Disable Code with NOPs

Disabling code with NOPs is one of the most straightforward patching techniques, and this exercise demonstrates how effective it can be for removing unwanted functionality.

When you examine the binary in Binary Ninja, you'll find the `call print_secret` instruction in the `main` function. This instruction is typically 5 bytes long on x64: one byte for the `call` opcode (`E8`) and four bytes for the relative offset to the `print_secret` function. The exact encoding might be something like `E8 12 34 56 78`, where `12 34 56 78` is the little-endian relative offset.

To disable this function call, you need to replace all 5 bytes with NOP instructions. The NOP instruction is encoded as `90`, so you'll replace `E8 12 34 56 78` with `90 90 90 90 90`. When execution reaches this location, it will execute five NOPs in sequence—doing nothing five times—and then continue to the next instruction. The `print_secret` function is never called, so "Secret message!" is never printed.

Open the binary in a hex editor, find the `call` instruction (you can search for the byte pattern or navigate to the file offset shown in Binary Ninja), and replace the 5 bytes with `90 90 90 90 90`. Save the file and run it. You should see only "Normal message" printed, confirming that the call to `print_secret` has been successfully disabled.

This technique is incredibly useful for removing anti-debugging code, disabling telemetry or logging functions, or bypassing checks without needing to understand what the function actually does. If you encounter a suspicious function call and want to see what happens when it's not executed, just NOP it out and observe the results.

### Solution 3: Create a Simple Inline Hook

Creating an inline hook is more complex than the previous exercises, but it demonstrates the power of runtime code modification and gives you a foundation for more advanced hooking techniques.

When you set a breakpoint at your target function in x64dbg and examine the first few instructions, you'll typically see a function prologue like:

```
push rbp
mov rbp, rsp
sub rsp, 0x20
```

To install an inline hook, you need to overwrite the first 5 bytes (enough for a `jmp` instruction) with a jump to your hook code. The `jmp` instruction with a relative 32-bit offset is encoded as `E9 XX XX XX XX`. You can use x64dbg's "Assemble" feature to do this: right-click on the first instruction, select "Assemble", and enter `jmp <address_of_your_hook>`.

For this exercise, you can create your hook code in an unused region of memory. In x64dbg, go to the Memory Map, find a region with execute permissions, and navigate to an unused area. Write your hook code there. A simple hook might look like:

```
; Your hook code
push rax
mov rax, 0x1234  ; Some custom value
pop rax
; Now jump back to the original function (after the overwritten bytes)
jmp original_function+5
```

When you modify the first instruction of the target function to `jmp your_hook_address`, execution will redirect to your hook code whenever the function is called. Your hook executes, performs whatever custom logic you want, and then jumps back to the original function at offset +5 (skipping the overwritten bytes).

To make this hook call the original function properly, you'd need to create a trampoline that executes the original 5 bytes you overwrote (the `push rbp; mov rbp, rsp` or whatever was there) and then jumps to `original_function+5`. This ensures the original function's prologue executes correctly.

When you test your hook, you should see your custom code execute whenever the target function is called. This demonstrates the fundamental technique behind game hacks, API monitors, and many other tools that intercept function calls at runtime.

## Summary

You now know how to patch binaries. You can:

- Modify instructions
- Use NOPs to disable code
- Flip conditional jumps
- Create inline hooks
- Hook the IAT
- Test and verify patches

In the next lesson, you'll learn about DLLs and how to hook game functions.
