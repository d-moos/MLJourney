# Lesson 2: Commercial Protectors – Analyzing VMProtect and Themida

## Overview

**Commercial protectors** like VMProtect and Themida use advanced techniques to protect binaries:
- **VM-based virtualization**: Code is compiled to bytecode and executed in a virtual machine
- **Code mutation**: Code changes every time it's executed
- **Anti-debugging**: Multiple anti-debugging techniques
- **Anti-VM**: Multiple anti-VM techniques
- **Obfuscation**: Complex control flow and data obfuscation

Understanding these protectors helps you analyze protected binaries.

## What You'll Learn

By the end of this lesson, you will understand:

- **How VMProtect works**
- **How Themida works**
- **VM-based virtualization concepts**
- **How to identify protected code**
- **Strategies for analyzing protected binaries**
- **Limitations of analysis tools**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1 of this course
- Understand x86-64 assembly
- Be comfortable with advanced reverse engineering

## VMProtect

**VMProtect** is one of the most sophisticated commercial software protection systems available. It uses VM-based virtualization to transform your code into a custom bytecode that's interpreted by a virtual machine embedded in the binary. This makes reverse engineering extremely difficult because the original x86/x64 instructions are completely replaced with a proprietary instruction set that's unique to each protected binary.

### How VMProtect Works

VMProtect's protection process involves several stages that fundamentally transform how your code executes.

**Code Selection** is the first step. When you use VMProtect, you don't protect the entire binary—instead, you select specific functions or code sections to protect. This is important because VM-based protection has significant performance overhead (typically 10-100x slower than native code), so you only protect critical functions like license checks, anti-cheat code, or cryptographic operations. You mark functions for protection using special markers in your source code or by selecting them in the VMProtect GUI.

**Compilation** is where the magic happens. VMProtect takes the selected functions and compiles them to a custom bytecode. This isn't just obfuscation—the original x86/x64 instructions are completely replaced. For example, a simple `add eax, ebx` might become a sequence of 20+ VM instructions that perform the same operation using the VM's virtual registers and instruction set. The bytecode is unique to each build, so even if you reverse engineer one protected binary, your knowledge doesn't transfer to another.

**VM Implementation** involves embedding a complete virtual machine interpreter into your binary. This VM includes a virtual CPU with its own registers, a virtual stack, and an instruction decoder/dispatcher. The VM is typically several thousand lines of highly obfuscated code that's extremely difficult to understand. Different versions of VMProtect use different VM architectures, and the VM itself is often protected with additional obfuscation layers.

**Execution** happens at runtime. When the program reaches a protected function, instead of executing native x86/x64 code, it enters the VM interpreter. The VM reads bytecode instructions one by one, decodes them, and executes the corresponding operations using its virtual registers and stack. From the CPU's perspective, it's executing the VM interpreter code; the actual logic of your function is hidden in the bytecode data.

### Identifying VMProtect

Recognizing VMProtect-protected binaries requires looking for several characteristic signs that distinguish them from normal or lightly-obfuscated binaries.

**Unusual entry point code** is often the first indicator. When you open a VMProtect-protected binary in a disassembler, the entry point doesn't look like normal compiler-generated code. Instead, you'll see highly obfuscated initialization code that sets up the VM environment. This code often includes anti-debugging checks, VM detection, and initialization of the VM's data structures.

**Large data sections** containing VM bytecode are another telltale sign. VMProtect stores the bytecode for protected functions in data sections (often `.vmp0`, `.vmp1`, etc.). These sections are typically large (hundreds of KB to several MB) and contain what appears to be random data—this is the encrypted/encoded bytecode. If you see large data sections with high entropy (randomness), it's likely VMProtect.

**Complex control flow** is characteristic of the VM interpreter itself. If you navigate to a protected function, you'll see an extremely complex control flow graph with hundreds of basic blocks, many indirect jumps, and a tangled web of connections. This is the VM's dispatcher and instruction handlers. Normal code, even obfuscated code, doesn't have this level of complexity.

**Anti-debugging and anti-VM checks** are heavily used by VMProtect. The protector includes dozens of different techniques to detect debuggers, virtual machines, and analysis tools. You'll see checks for `IsDebuggerPresent`, PEB.BeingDebugged, timing checks, hardware breakpoint detection, VM detection via CPUID, and many others. These checks are often integrated into the VM itself, making them harder to bypass.

**Suspicious imports** can also indicate VMProtect. The protector often imports functions like `VirtualProtect` (for changing memory permissions), `GetProcAddress` (for dynamic import resolution), and various anti-debugging APIs. However, VMProtect also hides imports by resolving them dynamically, so you might see fewer imports than expected.

### Analyzing VMProtect

Analyzing VMProtect-protected code is one of the most challenging tasks in reverse engineering, requiring a systematic approach and significant time investment.

**Identify protected functions** by looking for VM entry points. These are locations where the code transitions from normal x86/x64 execution to VM execution. You'll see the code saving the CPU state (registers), setting up the VM context, and jumping to the VM interpreter. Mark these locations because they're the boundaries between native and virtualized code.

**Dump the VM bytecode** by extracting it from memory or from the binary's data sections. The bytecode is often encrypted or encoded, so you may need to let the program run until the VM decrypts it, then dump it from memory. Tools like Scylla or manual memory dumping in x64dbg can help. The bytecode is typically stored as a stream of bytes that the VM interpreter reads sequentially.

**Reverse engineer the VM** by analyzing the interpreter code to understand the instruction set. This is extremely time-consuming. You need to identify the VM's dispatcher (the code that reads the next bytecode instruction), the instruction handlers (the code that executes each instruction type), and the virtual registers. Create a table mapping bytecode opcodes to their operations (e.g., opcode 0x42 = virtual ADD, opcode 0x73 = virtual PUSH). This often requires dynamic analysis, tracing through the VM execution and observing how different bytecode values affect the VM state.

**Decompile the bytecode** by converting it back to pseudocode or high-level operations. Once you understand the instruction set, you can write a disassembler that reads the bytecode and outputs something like "VPUSH vReg0; VPUSH vReg1; VADD; VPOP vReg2". Some researchers have created automated tools for this, but they're often specific to particular VMProtect versions.

**Analyze the decompiled code** to understand what the protected function actually does. Even after decompiling the bytecode, you still need to understand the logic. The VM adds significant complexity, so a simple license check might become hundreds of VM instructions. Focus on the high-level behavior rather than getting lost in the details of every VM instruction.

## Themida

**Themida** is another heavyweight commercial protector, developed by Oreans Technologies. It's similar to VMProtect in many ways but has its own unique features and protection techniques. Themida is known for being even more aggressive with anti-debugging and anti-analysis techniques, making it one of the most difficult protectors to analyze.

### How Themida Works

Themida's protection mechanism combines multiple layers of defense to create an extremely hostile environment for reverse engineers.

**Code Protection** works similarly to VMProtect—you select which functions to protect, and Themida transforms them. However, Themida offers multiple protection modes: virtualization (like VMProtect), mutation (code transformation that preserves functionality but changes structure), and combination modes that use both.

**Virtualization** in Themida uses a custom VM similar to VMProtect, but with a different architecture and instruction set. Themida's VM is known for being particularly complex, with multiple layers of indirection and encryption. The VM changes between versions, so techniques that work on one version may not work on another.

**Anti-Analysis** is where Themida really shines (or, from a reverser's perspective, where it becomes a nightmare). Themida includes an extensive array of anti-debugging techniques: it checks for debuggers using dozens of methods, detects virtual machines, monitors for analysis tools, checks for hooks, validates code integrity, and even includes anti-emulation techniques. These checks are scattered throughout the code and integrated into the VM, making them difficult to bypass comprehensively.

**Code Mutation** is a unique feature where Themida transforms code at runtime. The code in the binary isn't the code that actually executes—instead, Themida decrypts and mutates it during execution. This means the code changes each time it runs, and the code in the binary file doesn't match the code in memory. This defeats static analysis and makes memory dumping less effective.

### Identifying Themida

Themida has distinctive characteristics that help you identify it, though it tries hard to hide its presence.

**Unusual entry point code** in Themida is even more obfuscated than VMProtect. The entry point is often a maze of anti-debugging checks, junk code, and VM initialization. Themida also uses techniques like stolen bytes (moving the original entry point code elsewhere) and fake entry points to confuse analysts.

**Large data sections** are present, similar to VMProtect, containing encrypted code and VM bytecode. Themida often uses sections with names like `.themida` or custom section names. The data has very high entropy due to encryption.

**Complex control flow** is extreme in Themida. The protector uses control flow flattening, opaque predicates, and other obfuscation techniques on top of the VM, creating control flow graphs that are nearly impossible to understand visually.

**Anti-debugging and anti-VM checks** are pervasive. Themida is notorious for its aggressive anti-debugging. It uses timing checks, exception-based debugging detection, hardware breakpoint detection, software breakpoint detection (scanning for 0xCC bytes), VM detection, and even checks for specific analysis tools by name. These checks are constantly running, not just at startup.

**Suspicious imports** are often hidden. Themida resolves most imports dynamically to hide what APIs it uses. You might see very few imports in the import table, with most functions resolved at runtime using custom import resolution code.

### Analyzing Themida

Analyzing Themida requires all the techniques used for VMProtect, plus additional strategies to deal with its more aggressive protections.

**Identify protected functions** by finding VM entry points and mutation points. Themida's entry points are harder to spot due to additional obfuscation, but they still have characteristic patterns. Look for code that sets up a VM context or decrypts code sections.

**Bypass anti-debugging** before attempting deeper analysis. Themida's anti-debugging is so aggressive that you can't effectively analyze the binary while it's detecting your debugger. Use tools like ScyllaHide or TitanHide to hide your debugger, or patch out the anti-debugging checks (though Themida has integrity checks that detect patching). Some analysts use custom debuggers or emulators that Themida doesn't detect.

**Dump the VM bytecode** from memory after it's been decrypted. Themida encrypts the bytecode and only decrypts it during execution, so you need to let the program run, then dump the decrypted bytecode from memory. This often requires bypassing anti-debugging first.

**Reverse engineer the VM** using the same techniques as VMProtect, but expect it to be more complex. Themida's VM has more instructions, more layers of indirection, and more obfuscation. You'll need to spend significant time tracing through the VM interpreter to understand the instruction set.

**Decompile the bytecode** once you understand the VM. This is the same process as VMProtect but may require custom tools specific to Themida's VM architecture. Some researchers have published Themida devirtualization tools, but they're often version-specific.

## VM-Based Virtualization Concepts

Understanding the general architecture of VM-based protection helps you analyze any VM-protected binary, whether it's VMProtect, Themida, or a custom protector.

### VM Architecture

A typical software protection VM consists of several key components that work together to execute virtualized code.

**Registers** in the VM are virtual registers that don't correspond to real CPU registers. A VM might have 8, 16, or even 32 virtual registers (often called vR0, vR1, etc.). These are typically stored in memory (often on the stack or in a dedicated VM context structure) rather than in real CPU registers. The VM interpreter loads values from these virtual registers into real CPU registers when needed, performs operations, and stores the results back.

**Memory** in the VM can refer to either the real process memory or a separate virtual memory space. Simple VMs just use the process's normal memory, while sophisticated VMs create a separate virtual address space with its own memory layout. The VM provides instructions to read and write memory, translating virtual addresses to real addresses as needed.

**Instruction Set** is the heart of the VM. Each VM has a custom instruction set designed specifically for that protector. The instruction set includes opcodes (operation codes) that specify what operation to perform, and operands that specify what data to operate on. For example, a VM might have an instruction encoded as `[opcode: 0x42] [dest: vR0] [src1: vR1] [src2: vR2]` meaning "add vR1 and vR2, store result in vR0".

**Interpreter** is the code that executes VM instructions. It typically has a fetch-decode-execute loop: fetch the next instruction from the bytecode stream, decode it to determine what operation to perform, and execute the operation by calling the appropriate handler. The interpreter is usually implemented as a large switch statement or jump table that dispatches to different handlers based on the opcode.

### VM Instruction Set

While each VM has a unique instruction set, they typically include similar categories of instructions that mirror the capabilities of real CPUs.

**Load/Store** instructions move data between virtual registers and memory. For example, VLOAD might load a value from memory into a virtual register, while VSTORE writes a virtual register to memory. These instructions often include addressing modes like immediate, register-indirect, and offset-based addressing.

**Arithmetic** instructions perform mathematical operations. VADD adds two virtual registers, VSUB subtracts, VMUL multiplies, and VDIV divides. Some VMs also include more complex operations like modulo, shift, and rotate. These instructions typically take two source operands and one destination operand.

**Logic** instructions perform bitwise operations. VAND performs bitwise AND, VOR performs OR, VXOR performs XOR, and VNOT performs NOT. These are essential for implementing conditional logic and bit manipulation.

**Control Flow** instructions change the execution flow. VJMP performs an unconditional jump to a new bytecode location, VJCC performs a conditional jump based on flags or register values, VCALL calls a subroutine (pushing a return address), and VRET returns from a subroutine. These instructions are what make the VM Turing-complete.

**I/O** instructions interact with the outside world. These might include instructions to call native functions (transitioning from VM execution back to native x86/x64 execution), read/write specific memory locations, or perform system calls. Some VMs also include instructions for cryptographic operations, string manipulation, or other high-level operations.

## Exercises

### Exercise 1: Identify Protected Code

**Objective**: Learn to identify VMProtect/Themida protected code.

**Steps**:
1. Find a binary protected with VMProtect or Themida
2. Open it in Binary Ninja
3. Identify signs of protection
4. Document your findings

**Verification**: You should be able to identify protected code.

### Exercise 2: Analyze Protected Code

**Objective**: Learn to analyze protected code.

**Steps**:
1. Take a protected binary
2. Identify protected functions
3. Bypass anti-debugging
4. Dump the VM bytecode
5. Analyze the bytecode

**Verification**: You should be able to extract and analyze bytecode.

### Exercise 3: Reverse Engineer the VM

**Objective**: Learn to reverse engineer the VM.

**Steps**:
1. Extract the VM bytecode
2. Identify the VM instruction set
3. Write a disassembler for the VM
4. Disassemble the bytecode
5. Understand the bytecode

**Verification**: You should be able to disassemble and understand bytecode.

## Solutions

### Solution 1: Identify Protected Code

Here's a comprehensive guide to identifying VMProtect or Themida protected code in a binary.

**Step-by-Step Identification Process:**

1. **Initial Triage with Detect It Easy (DIE):**
   ```bash
   diec.exe suspicious_binary.exe
   ```

   DIE will often identify VMProtect or Themida directly. Look for output like:
   ```
   Protector: VMProtect 3.x
   Compiler: Microsoft Visual C++ 2019
   ```

2. **Analyze Entry Point in Binary Ninja:**

   Open the binary in Binary Ninja and navigate to the entry point. You'll see characteristic patterns:

   **VMProtect Entry Point:**
   ```asm
   ; Highly obfuscated initialization
   push rbp
   mov rbp, rsp
   sub rsp, 0x1000

   ; Anti-debugging checks
   call check_debugger
   test eax, eax
   jne exit_process

   ; VM initialization
   lea rax, [vm_context]
   call init_vm

   ; Jump to VM interpreter
   jmp vm_dispatcher
   ```

   **Themida Entry Point:**
   ```asm
   ; Even more obfuscated with stolen bytes
   db 0xEB, 0x10  ; jmp short +0x10 (obfuscated jump)
   db 0x90, 0x90  ; nops (junk code)

   ; Anti-debugging
   mov rax, gs:[0x60]  ; PEB
   movzx eax, byte ptr [rax+0x2]  ; BeingDebugged
   test al, al
   jne anti_debug_detected
   ```

3. **Check Section Headers:**

   Use PE-bear or Binary Ninja to examine sections:

   **VMProtect Sections:**
   ```
   .text    - Normal code section
   .vmp0    - VM bytecode (high entropy ~7.9)
   .vmp1    - VM data (high entropy ~7.8)
   .rdata   - Read-only data
   ```

   **Themida Sections:**
   ```
   .text     - Normal code section
   .themida  - Protected code (high entropy ~7.9)
   .data     - Data section
   ```

4. **Analyze Entropy:**

   Use Python with pefile:
   ```python
   import pefile

   pe = pefile.PE('suspicious_binary.exe')
   for section in pe.SECTIONS:
       name = section.Name.decode().rstrip('\x00')
       entropy = section.get_entropy()
       print(f"{name}: {entropy:.2f}")
   ```

   High entropy (>7.5) in code sections indicates encryption/packing.

5. **Check Import Table:**

   Protected binaries have sparse imports:
   ```
   KERNEL32.dll:
     - LoadLibraryA
     - GetProcAddress
     - VirtualProtect
     - VirtualAlloc
   ```

   Most imports are resolved dynamically to hide capabilities.

6. **Analyze Control Flow:**

   In Binary Ninja, look at the control flow graph of the entry point function. Protected code shows:
   - Hundreds of basic blocks
   - Extremely complex graph with many indirect jumps
   - No clear structure (unlike normal compiler-generated code)

**Complete Identification Report:**

```
=== Protection Analysis Report ===

Binary: suspicious_binary.exe
SHA256: [hash]

Protection Detected: VMProtect 3.5.1

Evidence:
1. DIE Detection: VMProtect 3.x
2. Section Analysis:
   - .vmp0 section present (entropy: 7.92)
   - .vmp1 section present (entropy: 7.85)
3. Entry Point: Highly obfuscated with VM initialization
4. Import Table: Sparse (only 12 imports, mostly dynamic resolution)
5. Control Flow: Extremely complex (347 basic blocks in entry function)
6. Anti-Analysis:
   - IsDebuggerPresent check at 0x401234
   - PEB.BeingDebugged check at 0x401250
   - Timing check at 0x401270

Recommendation: Requires advanced analysis techniques (VM reversing, anti-debug bypassing)
```

### Solution 2: Analyze Protected Code

Here's a complete walkthrough of analyzing VMProtect-protected code, including bypassing anti-debugging and dumping bytecode.

**Step 1: Bypass Anti-Debugging**

VMProtect uses multiple anti-debugging techniques. Here's how to bypass them:

**Using ScyllaHide (x64dbg plugin):**
```
1. Install ScyllaHide plugin for x64dbg
2. Open x64dbg
3. Plugins → ScyllaHide → Options
4. Enable all anti-debugging options:
   ☑ PEB.BeingDebugged
   ☑ PEB.NtGlobalFlag
   ☑ PEB.HeapFlags
   ☑ NtQueryInformationProcess
   ☑ NtSetInformationThread
   ☑ NtQuerySystemInformation
   ☑ OutputDebugString
   ☑ GetTickCount
   ☑ NtClose
5. Load the protected binary
```

**Manual Patching (alternative method):**
```asm
; Find anti-debug check:
call IsDebuggerPresent
test eax, eax
jne exit_process

; Patch to:
call IsDebuggerPresent
xor eax, eax        ; Force EAX to 0 (not debugging)
nop                 ; Replace jne with nop
nop
```

**Step 2: Identify Protected Functions**

Protected functions have characteristic VM entry points:

```asm
; Normal function:
push rbp
mov rbp, rsp
sub rsp, 0x20
; ... normal code ...

; VMProtect protected function:
push rbp
mov rbp, rsp
sub rsp, 0x1000
lea rax, [vm_context]
mov [rax], rbp
mov [rax+8], rsp
call vm_enter
; ... VM bytecode execution ...
call vm_exit
mov rsp, rbp
pop rbp
ret
```

**Step 3: Dump VM Bytecode**

The bytecode is stored in the `.vmp0` section. Here's how to dump it:

**Using x64dbg:**
```
1. Open the binary in x64dbg
2. Go to Memory Map tab
3. Find the .vmp0 section
4. Right-click → Dump Memory to File
5. Save as "bytecode.bin"
```

**Using Python script:**
```python
import pefile

pe = pefile.PE('protected.exe')

# Find .vmp0 section
for section in pe.SECTIONS:
    if b'.vmp0' in section.Name:
        # Extract bytecode
        bytecode = section.get_data()

        with open('bytecode.bin', 'wb') as f:
            f.write(bytecode)

        print(f"Dumped {len(bytecode)} bytes of VM bytecode")
        print(f"Entropy: {section.get_entropy():.2f}")
```

**Step 4: Analyze the Bytecode**

The bytecode is encrypted. You need to let the VM decrypt it first:

```
1. Set breakpoint at VM entry: bp vm_enter
2. Run until breakpoint
3. Step through VM initialization
4. Find where bytecode is decrypted
5. Set memory breakpoint on bytecode region
6. When decryption completes, dump the decrypted bytecode
```

**Decrypted Bytecode Analysis:**
```python
# Analyze decrypted bytecode
with open('bytecode_decrypted.bin', 'rb') as f:
    bytecode = f.read()

# Look for patterns
print("First 100 bytes:")
print(' '.join(f'{b:02x}' for b in bytecode[:100]))

# Identify potential opcodes
opcodes = {}
for byte in bytecode:
    opcodes[byte] = opcodes.get(byte, 0) + 1

print("\nMost common bytes (potential opcodes):")
for opcode, count in sorted(opcodes.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"0x{opcode:02x}: {count} occurrences")
```

### Solution 3: Reverse Engineer the VM

Here's a complete guide to reverse engineering the VMProtect VM instruction set.

**Step 1: Identify the VM Dispatcher**

The dispatcher is the core loop that reads and executes bytecode instructions:

```asm
vm_dispatcher:
    ; Read next opcode
    movzx eax, byte ptr [rsi]    ; RSI points to bytecode
    inc rsi                       ; Move to next byte

    ; Dispatch to handler
    lea rbx, [handler_table]
    mov rax, [rbx + rax*8]       ; Get handler address
    jmp rax                       ; Jump to handler
```

**Step 2: Map Opcodes to Handlers**

Create a table mapping opcodes to handler addresses:

```python
# Trace VM execution and log opcode → handler mappings
opcode_handlers = {}

# Example from tracing:
opcode_handlers = {
    0x01: 0x401000,  # VPUSH handler
    0x02: 0x401100,  # VPOP handler
    0x03: 0x401200,  # VADD handler
    0x04: 0x401300,  # VSUB handler
    0x05: 0x401400,  # VMUL handler
    # ... etc
}
```

**Step 3: Reverse Engineer Each Handler**

Analyze each handler to understand what operation it performs:

**Example: VADD Handler**
```asm
handler_vadd:
    ; Pop two values from virtual stack
    mov rax, [vsp]
    sub vsp, 8
    mov rbx, [vsp]
    sub vsp, 8

    ; Add them
    add rax, rbx

    ; Push result
    mov [vsp], rax
    add vsp, 8

    ; Return to dispatcher
    jmp vm_dispatcher
```

This is a virtual ADD instruction: pop two values, add them, push result.

**Step 4: Write a Disassembler**

Create a disassembler that converts bytecode to readable instructions:

```python
class VMDisassembler:
    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.pc = 0

        # Opcode definitions (from reverse engineering)
        self.opcodes = {
            0x01: ('VPUSH', 1),  # (name, operand_bytes)
            0x02: ('VPOP', 0),
            0x03: ('VADD', 0),
            0x04: ('VSUB', 0),
            0x05: ('VMUL', 0),
            0x10: ('VLOAD', 1),
            0x11: ('VSTORE', 1),
            0x20: ('VJMP', 4),
            0x21: ('VJCC', 4),
        }

    def disassemble(self):
        """Disassemble the entire bytecode."""
        instructions = []

        while self.pc < len(self.bytecode):
            opcode = self.bytecode[self.pc]
            self.pc += 1

            if opcode not in self.opcodes:
                instructions.append(f"{self.pc-1:04x}: UNKNOWN 0x{opcode:02x}")
                continue

            name, operand_bytes = self.opcodes[opcode]

            # Read operands
            operands = []
            for _ in range(operand_bytes):
                operands.append(self.bytecode[self.pc])
                self.pc += 1

            # Format instruction
            if operands:
                operand_str = ', '.join(f'0x{op:02x}' for op in operands)
                instructions.append(f"{self.pc-1-operand_bytes:04x}: {name} {operand_str}")
            else:
                instructions.append(f"{self.pc-1:04x}: {name}")

        return instructions

# Use the disassembler
with open('bytecode_decrypted.bin', 'rb') as f:
    bytecode = f.read()

disasm = VMDisassembler(bytecode)
instructions = disasm.disassemble()

for instr in instructions[:50]:  # Print first 50 instructions
    print(instr)
```

**Example Output:**
```
0000: VPUSH 0x0a
0002: VPUSH 0x14
0004: VADD
0005: VSTORE 0x00
0007: VLOAD 0x00
0009: VPUSH 0x02
000b: VMUL
000c: VJCC 0x0020
```

**Step 5: Decompile to Pseudocode**

Convert the disassembled VM instructions to higher-level pseudocode:

```python
def decompile(instructions):
    """Convert VM instructions to pseudocode."""
    stack = []
    variables = {}
    pseudocode = []

    for instr in instructions:
        parts = instr.split()
        opcode = parts[1]

        if opcode == 'VPUSH':
            value = parts[2]
            stack.append(value)
        elif opcode == 'VADD':
            b = stack.pop()
            a = stack.pop()
            result = f"({a} + {b})"
            stack.append(result)
        elif opcode == 'VSTORE':
            var = parts[2]
            value = stack.pop()
            variables[var] = value
            pseudocode.append(f"var_{var} = {value}")
        # ... handle other opcodes

    return pseudocode
```

This process reveals the original logic hidden by the VM protection, allowing you to understand what the protected code actually does.

## Summary

You now understand commercial protectors. You can:

- Identify VMProtect and Themida protected code
- Analyze protected binaries
- Understand VM-based virtualization
- Reverse engineer VMs
- Extract and analyze bytecode

In the next lesson, you'll learn about VM unpacking.
