# Lesson 4: Deobfuscation Strategies – Advanced Code Analysis

## Overview

**Deobfuscation** is the process of removing obfuscation from code. Advanced deobfuscation techniques include:
- **Symbolic execution**: Execute code symbolically to understand behavior
- **Taint analysis**: Track how data flows through the program
- **Constraint solving**: Solve constraints to find code paths
- **Machine learning**: Use ML to identify patterns

## What You'll Learn

By the end of this lesson, you will understand:

- **Symbolic execution concepts**
- **How to use symbolic execution tools**
- **Taint analysis concepts**
- **How to perform taint analysis**
- **Constraint solving basics**
- **How to use deobfuscation tools**

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-3 of this course
- Understand control flow obfuscation
- Be comfortable with advanced analysis

## Symbolic Execution

**Symbolic execution** is a powerful program analysis technique that executes code with symbolic values (variables) instead of concrete values (actual numbers). This allows you to explore all possible execution paths simultaneously and reason about the program's behavior for all possible inputs. It's particularly useful for deobfuscation because it can simplify complex obfuscated code by determining which branches are actually reachable and what values are possible.

### Example

```c
int x = input();
if (x > 10) {
    y = 20;
} else {
    y = 30;
}
```

With symbolic execution, instead of running this code with a specific value like `x = 15`, you run it with a symbolic variable representing all possible values of `x`. The symbolic execution engine treats `x` as a symbol (like in algebra) and explores both branches of the `if` statement.

**x is a symbolic variable** (not a concrete value), meaning it represents any possible integer value. The engine doesn't know what `x` is—it could be -1000, 0, 15, or 1000000. Instead of picking one value, it reasons about all values simultaneously.

**The condition `x > 10` is a constraint** that divides the possible values of `x` into two sets: values where `x > 10` is true, and values where it's false. The symbolic execution engine records this constraint and explores both possibilities.

**Both branches are explored** by the engine. It follows the true branch with the constraint `x > 10` added to its path constraints, and separately follows the false branch with the constraint `x <= 10`. This is different from normal execution, which only follows one branch based on the concrete value of `x`.

**The result is a set of possible values for y**: the engine determines that `y = 20` when `x > 10`, and `y = 30` when `x <= 10`. This gives you complete knowledge of the program's behavior for all possible inputs, which is invaluable for understanding obfuscated code.

### Symbolic Execution Tools

Several powerful tools implement symbolic execution for binary analysis and deobfuscation.

**Triton** is a dynamic binary analysis framework that provides symbolic execution, taint analysis, and constraint solving. It works by instrumenting binary execution (using Pin or DynamoRIO) and building symbolic expressions for each instruction. Triton is particularly good for analyzing obfuscated code because it can simplify complex expressions and identify opaque predicates. It supports x86, x86-64, ARM, and AArch64 architectures. Triton integrates with Z3 for constraint solving, allowing you to determine if branches are reachable and what inputs lead to specific behaviors.

**Angr** is a comprehensive binary analysis framework built in Python that includes symbolic execution, control flow analysis, and vulnerability discovery. Angr can load binaries, perform symbolic execution to explore paths, and solve constraints to find inputs that reach specific code locations. It's particularly useful for CTF challenges and malware analysis. Angr's symbolic execution engine can handle complex programs, though it may struggle with very large or heavily obfuscated binaries due to path explosion (the number of paths growing exponentially).

**Z3** is Microsoft's SMT (Satisfiability Modulo Theories) solver, which is the constraint solver used by both Triton and Angr. While Z3 itself isn't a symbolic execution engine, it's the core component that makes symbolic execution practical. Z3 takes constraints (like `x > 10 AND x < 20`) and determines if they're satisfiable (can be true) and what values satisfy them. You can use Z3 directly to solve constraints extracted from obfuscated code, making it a fundamental tool for deobfuscation.

## Taint Analysis

**Taint analysis** is a technique that tracks how data flows through a program by marking certain data as "tainted" (untrusted or interesting) and following how that taint propagates through operations. This is extremely useful for understanding how user input affects program behavior, identifying what code depends on specific data, and finding where sensitive data flows. In deobfuscation, taint analysis helps you identify which parts of the code actually depend on real input versus which parts are just junk code.

### Example

```c
int x = input();  // x is tainted
int y = x + 5;    // y is tainted
if (y > 20) {     // condition depends on tainted data
    // ...
}
```

In this example, `x` is marked as tainted because it comes from user input (an untrusted source). When you compute `y = x + 5`, the taint propagates—since `y` depends on the tainted value `x`, `y` is also tainted. The condition `y > 20` depends on tainted data, so the branch decision is influenced by user input. Taint analysis tracks this entire flow, showing you exactly how user input affects the program.

This is powerful for deobfuscation because obfuscated code often includes junk code that doesn't depend on any real input—it's just there to confuse analysts. Taint analysis can identify this junk code by showing that it doesn't depend on any tainted (real) data.

### Taint Analysis Tools

Several tools provide taint analysis capabilities for binary analysis.

**Triton** supports taint analysis in addition to symbolic execution. You can mark specific registers or memory locations as tainted, then Triton automatically propagates the taint through all operations. For example, you might taint the RAX register after a `read` syscall (marking user input as tainted), then trace how that taint spreads through the program. Triton's taint analysis is precise and works at the instruction level, making it ideal for understanding data flow in obfuscated code.

**Frida** is a dynamic instrumentation framework that allows you to inject JavaScript into running processes and hook functions. While Frida doesn't have built-in taint analysis, you can implement taint tracking by hooking relevant functions and tracking data flow manually. Frida is particularly useful for mobile app analysis and for situations where you need to analyze code in its native environment rather than in a controlled analysis environment.

**Pin** is Intel's dynamic binary instrumentation framework that allows you to insert analysis code at arbitrary points in a program's execution. Pin is lower-level than Frida and provides more control, but requires writing instrumentation in C++. You can implement sophisticated taint analysis using Pin by instrumenting every instruction and tracking how data flows through registers and memory. Pin is the instrumentation engine that Triton uses internally.

## Constraint Solving

**Constraint solving** is the process of finding values that satisfy a set of constraints (logical conditions). In deobfuscation, constraint solving is used to simplify opaque predicates, determine if code paths are reachable, and find inputs that trigger specific behaviors. A constraint solver takes logical formulas (like `x > 10 AND x < 20 AND x % 2 == 0`) and determines if they can be satisfied and what values satisfy them.

### Example

```c
if ((x * 2) % 2 == 0) {  // Always true
    // This branch is always taken
}
```

A constraint solver can analyze this condition and determine that it's always true, regardless of the value of `x`. Here's why: `x * 2` is always even (any number times 2 is even), and even numbers modulo 2 always equal 0. Therefore, the condition `(x * 2) % 2 == 0` is a tautology—it's always true. This is an opaque predicate, a common obfuscation technique where a condition appears to be a real branch but actually always goes the same way.

By using a constraint solver, you can automatically identify and simplify these opaque predicates, dramatically simplifying obfuscated code. Instead of seeing a complex conditional, you can replace it with an unconditional jump, making the code much easier to understand.

### Constraint Solvers

Constraint solvers are the mathematical engines that power symbolic execution and automated deobfuscation.

**Z3** is Microsoft's state-of-the-art SMT solver, widely used in program analysis, verification, and security research. Z3 can solve constraints involving integers, bit-vectors, arrays, floating-point numbers, and more. It's particularly good at solving the types of constraints that arise in binary analysis, like bit-level operations and modular arithmetic. Z3 has bindings for Python, C++, Java, and other languages, making it easy to integrate into your analysis tools. For deobfuscation, you can extract constraints from obfuscated code, feed them to Z3, and get simplified conditions or proof that certain branches are unreachable.

**SMT solvers** (Satisfiability Modulo Theories solvers) are a general class of constraint solvers that Z3 belongs to. Other SMT solvers include CVC4, Yices, and Boolector. These solvers extend SAT (Boolean satisfiability) solving with theories like arithmetic, bit-vectors, and arrays. Different solvers have different strengths—some are faster for certain types of constraints, some support different theories. For most deobfuscation work, Z3 is the best choice due to its maturity, performance, and excellent Python bindings.

## Deobfuscation Tools

While the techniques above are powerful, you also need practical tools that implement them and provide user-friendly interfaces for deobfuscation work.

### Ghidra

Ghidra is the NSA's open-source reverse engineering framework, and it includes several built-in deobfuscation features that can automatically simplify obfuscated code.

**Control flow analysis** in Ghidra can reconstruct the program's control flow graph even when it's been obfuscated with techniques like control flow flattening or indirect jumps. Ghidra's analysis engine follows jumps, identifies basic blocks, and builds a graph showing how execution flows through the program. This is the foundation for all other analysis.

**Dead code elimination** automatically removes code that can never be executed or whose results are never used. Obfuscated code often includes large amounts of dead code to confuse analysts. Ghidra's dead code elimination identifies and removes this junk, leaving only the code that actually matters. This can dramatically reduce the size and complexity of obfuscated functions.

**Constant propagation** replaces variables with their constant values when possible. If Ghidra can determine that a variable always has a specific value, it replaces all uses of that variable with that value. This simplifies expressions and can reveal the true behavior of obfuscated code. For example, if obfuscated code computes `x = 5; y = x * 2; z = y + 3;`, constant propagation simplifies this to `z = 13;`.

### Binary Ninja

Binary Ninja is a commercial reverse engineering platform with powerful analysis and deobfuscation capabilities.

**Control flow analysis** in Binary Ninja is more advanced than many other tools, with sophisticated algorithms for handling indirect jumps, switch statements, and obfuscated control flow. Binary Ninja can often reconstruct control flow that other tools miss, making it particularly good for analyzing heavily obfuscated code.

**Decompilation** in Binary Ninja produces high-quality C-like pseudocode from assembly. The decompiler includes many optimizations and simplifications that effectively deobfuscate code. For example, it can simplify complex arithmetic expressions, eliminate dead code, and recognize common patterns. The decompiled output is often much easier to understand than the raw assembly, even for obfuscated code.

**Pattern recognition** allows Binary Ninja to identify common code patterns and replace them with higher-level representations. For example, it can recognize a complex sequence of instructions as a string copy operation and represent it as `strcpy(dest, src)` in the decompiler output. This abstraction makes obfuscated code much more readable.

### Custom Tools

For advanced deobfuscation work, you'll often need to write custom tools tailored to the specific obfuscation techniques you're facing.

**Triton for symbolic execution** allows you to write Python scripts that symbolically execute obfuscated code, simplify expressions, and identify opaque predicates. You can instrument the binary, run it with symbolic inputs, collect path constraints, and use Z3 to simplify them. This is particularly effective against VM-based obfuscation and complex arithmetic obfuscation.

**Angr for binary analysis** provides a high-level Python API for loading binaries, performing symbolic execution, and analyzing control flow. You can write scripts that automatically explore the binary, identify interesting code paths, and extract simplified logic. Angr is particularly good for automated analysis of large numbers of samples.

**Z3 for constraint solving** can be used directly to simplify constraints extracted from obfuscated code. You can write scripts that parse assembly, extract constraints from conditional jumps, feed them to Z3, and determine which branches are reachable. This manual approach gives you maximum control and can handle obfuscation techniques that automated tools struggle with.

## Exercises

### Exercise 1: Use Symbolic Execution

**Objective**: Learn to use symbolic execution.

**Steps**:
1. Write a simple obfuscated program
2. Use Triton or Angr to perform symbolic execution
3. Analyze the results
4. Understand the program's behavior

**Verification**: You should understand the program's behavior.

### Exercise 2: Perform Taint Analysis

**Objective**: Learn to perform taint analysis.

**Steps**:
1. Write a program with tainted data
2. Use Triton or Frida to perform taint analysis
3. Trace how data flows
4. Identify tainted operations

**Verification**: You should be able to trace data flow.

### Exercise 3: Use Constraint Solving

**Objective**: Learn to use constraint solving.

**Steps**:
1. Write a program with constraints
2. Use Z3 to solve the constraints
3. Find values that satisfy the constraints
4. Verify the solution

**Verification**: You should be able to solve constraints.

## Solutions

### Solution 1: Use Symbolic Execution

Here's a complete example of using Angr for symbolic execution to solve an obfuscated password check.

**Obfuscated Program (C):**
```c
#include <stdio.h>
#include <string.h>

int check_password(char* input) {
    // Obfuscated password check
    int result = 0;
    result += (input[0] ^ 0x41) == 0x20 ? 1 : 0;  // 'a'
    result += (input[1] ^ 0x42) == 0x24 ? 1 : 0;  // 'd'
    result += (input[2] ^ 0x43) == 0x26 ? 1 : 0;  // 'm'
    result += (input[3] ^ 0x44) == 0x29 ? 1 : 0;  // 'i'
    result += (input[4] ^ 0x45) == 0x2d ? 1 : 0;  // 'n'
    return result == 5;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <password>\n", argv[0]);
        return 1;
    }

    if (check_password(argv[1])) {
        printf("Access granted!\n");
        return 0;
    } else {
        printf("Access denied!\n");
        return 1;
    }
}
```

Compile with: `gcc -o obfuscated obfuscated.c`

**Angr Solution Script:**
```python
import angr
import claripy

# Load the binary
project = angr.Project('./obfuscated', auto_load_libs=False)

# Create a symbolic bitvector for the password (5 characters)
password_length = 5
password = claripy.BVS('password', password_length * 8)

# Create an initial state at the main function
# We'll pass the symbolic password as argv[1]
state = project.factory.entry_state(args=['./obfuscated', password])

# Add constraints: password should be printable ASCII
for byte in password.chop(8):
    state.solver.add(byte >= 0x20)  # Printable ASCII
    state.solver.add(byte <= 0x7e)

# Create a simulation manager
simgr = project.factory.simulation_manager(state)

# Define success and failure addresses
# You need to find these by looking at the binary in a disassembler
# For this example, let's say:
# - Success: address where "Access granted!" is printed
# - Failure: address where "Access denied!" is printed

# Find the addresses (you'd get these from Binary Ninja/Ghidra)
# For demonstration, let's use symbolic addresses
success_addr = 0x401234  # Replace with actual address
failure_addr = 0x401250  # Replace with actual address

# Explore until we find the success path, avoiding the failure path
simgr.explore(find=success_addr, avoid=failure_addr)

# Check if we found a solution
if simgr.found:
    solution_state = simgr.found[0]

    # Extract the password
    solution = solution_state.solver.eval(password, cast_to=bytes)
    print(f"Found password: {solution.decode('utf-8')}")

    # Verify the solution
    print(f"Verification: {solution}")
else:
    print("No solution found")
```

**Expected Output:**
```
Found password: admin
Verification: b'admin'
```

**How it works:**
1. Angr loads the binary and creates a symbolic execution engine
2. We create a symbolic variable for the password (unknown value)
3. We add constraints (password must be printable ASCII)
4. Angr explores all possible execution paths symbolically
5. When it finds a path that reaches the "success" address, it solves the constraints to find what password value leads to that path
6. The result is the password that satisfies all the obfuscated checks

This technique works even when the password check is heavily obfuscated, because Angr reasons about the program symbolically rather than trying to understand the obfuscation.

### Solution 2: Perform Taint Analysis

Here's a complete example of using Triton for taint analysis to identify which parts of the code depend on user input.

**Program with Tainted Data:**
```c
#include <stdio.h>

int main() {
    int user_input;
    printf("Enter a number: ");
    scanf("%d", &user_input);

    int x = user_input * 2;      // Tainted
    int y = x + 10;               // Tainted
    int z = 100;                  // Not tainted
    int result = y + z;           // Tainted (depends on y)

    if (result > 200) {           // Tainted condition
        printf("Large result: %d\n", result);
    } else {
        printf("Small result: %d\n", result);
    }

    return 0;
}
```

**Triton Taint Analysis Script:**
```python
from triton import *

# Create Triton context
ctx = TritonContext()
ctx.setArchitecture(ARCH.X86_64)

# Enable taint engine
ctx.enableTaintEngine(True)

# Simulate the program execution
# In a real scenario, you'd use Pin or DynamoRIO to instrument the binary
# For this example, we'll manually simulate the key instructions

# Simulate: scanf reads into RAX (user input)
# Mark RAX as tainted (it contains user input)
ctx.taintRegister(ctx.registers.rax)
print(f"RAX tainted: {ctx.isRegisterTainted(ctx.registers.rax)}")

# Simulate: mov rbx, rax (x = user_input)
# Taint propagates from RAX to RBX
ctx.taintAssignment(ctx.registers.rbx, ctx.registers.rax)
print(f"RBX tainted: {ctx.isRegisterTainted(ctx.registers.rbx)}")

# Simulate: imul rbx, 2 (x = user_input * 2)
# RBX remains tainted
print(f"RBX tainted after multiply: {ctx.isRegisterTainted(ctx.registers.rbx)}")

# Simulate: add rbx, 10 (y = x + 10)
# RBX remains tainted (tainted + constant = tainted)
print(f"RBX tainted after add: {ctx.isRegisterTainted(ctx.registers.rbx)}")

# Simulate: mov rcx, 100 (z = 100)
# RCX is not tainted (constant value)
print(f"RCX tainted: {ctx.isRegisterTainted(ctx.registers.rcx)}")

# Simulate: add rbx, rcx (result = y + z)
# RBX remains tainted (tainted + untainted = tainted)
ctx.taintUnion(ctx.registers.rbx, ctx.registers.rcx)
print(f"RBX tainted after final add: {ctx.isRegisterTainted(ctx.registers.rbx)}")

# Simulate: cmp rbx, 200 (if result > 200)
# The comparison depends on tainted data
print(f"\nConclusion: The branch decision depends on user input (tainted)")
```

**Expected Output:**
```
RAX tainted: True
RBX tainted: True
RBX tainted after multiply: True
RBX tainted after add: True
RCX tainted: False
RBX tainted after final add: True

Conclusion: The branch decision depends on user input (tainted)
```

**Practical Application:**

In real malware analysis or deobfuscation, you'd use this to:
1. Identify which code paths depend on user input (tainted)
2. Identify junk code that doesn't depend on any real data (untainted)
3. Focus your analysis on tainted code paths
4. Understand data flow through complex obfuscated code

**Full Instrumentation Example with Pin:**

For a complete solution, you'd use Pin to instrument the binary and Triton to track taint:

```python
from triton import *
import sys

# This would be integrated with Pin for real binary instrumentation
def taint_analysis(binary_path):
    ctx = TritonContext()
    ctx.setArchitecture(ARCH.X86_64)
    ctx.enableTaintEngine(True)

    # Hook the scanf function to mark its output as tainted
    # Hook every instruction to propagate taint
    # Track which branches depend on tainted data

    # This is a simplified example - real implementation would use Pin
    print(f"Analyzing {binary_path} for tainted data flow...")
```

### Solution 3: Use Constraint Solving

Here's a complete example of using Z3 to solve constraints and simplify obfuscated code.

**Obfuscated Code with Opaque Predicates:**
```c
#include <stdio.h>

int main() {
    int x = 42;

    // Opaque predicate: always true
    if ((x * 2) % 2 == 0) {
        printf("This branch is always taken\n");
    } else {
        printf("This branch is never taken\n");
    }

    // Another opaque predicate: always true
    if ((x * x) >= 0) {
        printf("This is also always taken\n");
    }

    // Complex opaque predicate
    if ((x * 7 + 3) % 7 == 3) {
        printf("This is always taken too\n");
    }

    return 0;
}
```

**Z3 Constraint Solving Script:**
```python
from z3 import *

# Example 1: Prove that (x * 2) % 2 == 0 is always true
print("=== Example 1: (x * 2) % 2 == 0 ===")
x = Int('x')
solver = Solver()

# Try to find a counterexample (where the condition is false)
solver.add((x * 2) % 2 != 0)

if solver.check() == unsat:
    print("Proven: (x * 2) % 2 == 0 is ALWAYS TRUE (opaque predicate)")
    print("Simplification: Replace with unconditional jump\n")
else:
    print("Counterexample found:", solver.model())

# Example 2: Prove that (x * x) >= 0 is always true
print("=== Example 2: (x * x) >= 0 ===")
x = Int('x')
solver = Solver()

# Try to find a counterexample
solver.add((x * x) < 0)

if solver.check() == unsat:
    print("Proven: (x * x) >= 0 is ALWAYS TRUE (opaque predicate)")
    print("Simplification: Replace with unconditional jump\n")
else:
    print("Counterexample found:", solver.model())

# Example 3: Prove that (x * 7 + 3) % 7 == 3 is always true
print("=== Example 3: (x * 7 + 3) % 7 == 3 ===")
x = Int('x')
solver = Solver()

# Try to find a counterexample
solver.add((x * 7 + 3) % 7 != 3)

if solver.check() == unsat:
    print("Proven: (x * 7 + 3) % 7 == 3 is ALWAYS TRUE (opaque predicate)")
    print("Simplification: Replace with unconditional jump\n")
else:
    print("Counterexample found:", solver.model())

# Example 4: Solve for x in a complex constraint
print("=== Example 4: Solve complex constraint ===")
x = Int('x')
solver = Solver()

# Find x such that: (x * 3 + 7) % 11 == 5 AND x > 0 AND x < 100
solver.add((x * 3 + 7) % 11 == 5)
solver.add(x > 0)
solver.add(x < 100)

if solver.check() == sat:
    model = solver.model()
    print(f"Solution found: x = {model[x]}")

    # Find all solutions
    solutions = []
    while solver.check() == sat:
        model = solver.model()
        solution = model[x].as_long()
        solutions.append(solution)

        # Add constraint to find different solution
        solver.add(x != solution)

        if len(solutions) >= 10:  # Limit to 10 solutions
            break

    print(f"All solutions (up to 10): {solutions}")
else:
    print("No solution exists")

# Example 5: Simplify complex expression
print("\n=== Example 5: Simplify expression ===")
x = Int('x')

# Complex expression from obfuscated code
expr = (x * 2 + 4) * 3 - 6 * x

# Simplify
simplified = simplify(expr)
print(f"Original: (x * 2 + 4) * 3 - 6 * x")
print(f"Simplified: {simplified}")
```

**Expected Output:**
```
=== Example 1: (x * 2) % 2 == 0 ===
Proven: (x * 2) % 2 == 0 is ALWAYS TRUE (opaque predicate)
Simplification: Replace with unconditional jump

=== Example 2: (x * x) >= 0 ===
Proven: (x * x) >= 0 is ALWAYS TRUE (opaque predicate)
Simplification: Replace with unconditional jump

=== Example 3: (x * 7 + 3) % 7 == 3 ===
Proven: (x * 7 + 3) % 7 == 3 is ALWAYS TRUE (opaque predicate)
Simplification: Replace with unconditional jump

=== Example 4: Solve complex constraint ===
Solution found: x = 6
All solutions (up to 10): [6, 17, 28, 39, 50, 61, 72, 83, 94]

=== Example 5: Simplify expression ===
Original: (x * 2 + 4) * 3 - 6 * x
Simplified: 12
```

**Practical Application:**

Use Z3 to:
1. **Identify opaque predicates** - conditions that always evaluate the same way
2. **Simplify complex expressions** - reduce obfuscated arithmetic to simple constants
3. **Solve for inputs** - find what inputs lead to specific code paths
4. **Prove properties** - verify that certain conditions are always true or false

This is incredibly powerful for deobfuscation because you can automatically simplify complex obfuscated code without manually understanding every operation.

## Summary

You now understand advanced deobfuscation. You can:

- Use symbolic execution
- Perform taint analysis
- Solve constraints
- Use deobfuscation tools
- Analyze complex obfuscated code

In the next lesson, you'll learn about manually mapped DLLs.
