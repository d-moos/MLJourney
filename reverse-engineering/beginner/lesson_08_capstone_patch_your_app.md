# Lesson 8: Capstone Project – Reverse Engineer and Patch Your Own Application

## Overview

In this capstone project, you'll bring together everything you've learned. You'll:
1. Create a target application with a "secret" feature
2. Reverse engineer it to find the secret
3. Patch it to unlock the secret feature
4. Document your findings

This project demonstrates your mastery of the beginner course.

## Project Requirements

### Part 1: Create a Target Application

Create a Rust program with the following features:

1. **A password check**: The program asks for a password. If correct, it prints "Access granted!". If incorrect, it prints "Access denied!".

2. **A hidden feature**: There's a function that prints a secret message, but it's only called if a certain condition is met (e.g., a global variable is set to a specific value).

3. **Obfuscation** (optional): Make the password check slightly non-obvious (e.g., use bitwise operations, XOR, etc.).

Example:

```rust
fn main() {
    println!("Enter password: ");
    let mut password = String::new();
    std::io::stdin().read_line(&mut password).unwrap();
    
    if check_password(&password) {
        println!("Access granted!");
        print_secret();
    } else {
        println!("Access denied!");
    }
}

fn check_password(password: &str) -> bool {
    password.trim() == "secret123"
}

fn print_secret() {
    println!("You found the secret!");
}
```

### Part 2: Reverse Engineer the Application

1. Open your application in Binary Ninja
2. Find the password check function
3. Understand how it works
4. Find the secret function
5. Understand when it's called
6. Document your findings

### Part 3: Patch the Application

1. Modify the binary to:
   - Always pass the password check (even with wrong input)
   - Or always call the secret function
   - Or modify the password to something you know

2. Test the patched binary to verify it works

### Part 4: Document Your Work

Create a report that includes:

1. **Overview**: What the application does
2. **Reverse Engineering**: How you found the password check and secret function
3. **Analysis**: How the password check works
4. **Patching**: What you modified and how
5. **Results**: Screenshots or output showing the patched binary working
6. **Lessons Learned**: What you learned from this project

## Detailed Steps

### Step 1: Create the Target Application

```rust
fn main() {
    println!("=== Secret Application ===");
    println!("Enter the password: ");
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    
    if verify_password(input.trim()) {
        println!("Access granted!");
        unlock_secret();
    } else {
        println!("Access denied!");
    }
}

fn verify_password(password: &str) -> bool {
    // Simple check: password is "admin123"
    password == "admin123"
}

fn unlock_secret() {
    println!("\n=== SECRET MESSAGE ===");
    println!("Congratulations! You've unlocked the secret!");
    println!("The secret code is: REVERSING_MASTER");
}
```

Compile with: `rustc -O main.rs`

### Step 2: Reverse Engineer

1. Open `main.exe` in Binary Ninja
2. Find the `main` function
3. Look for the password check (string comparison)
4. Find the `unlock_secret` function
5. Understand the control flow

### Step 3: Patch

Option A: Modify the password check to always return true
- Find the `cmp` instruction that compares the password
- Change the conditional jump to always jump

Option B: Modify the condition to always call `unlock_secret`
- Find the `je` (jump if equal) instruction
- Change it to `jmp` (unconditional jump)

Option C: Modify the password in memory
- Find the password string in the binary
- Change it to something you know

### Step 4: Document

Write a report describing:
- What you found
- How you found it
- What you changed
- Why it worked

## Exercises

### Exercise 1: Reverse Engineer the Application

**Objective**: Find the password and secret function.

**Steps**:
1. Open the application in Binary Ninja
2. Find the `main` function
3. Identify the password check
4. Identify the secret function
5. Document your findings

**Verification**: You should be able to describe how the password check works and when the secret function is called.

### Exercise 2: Patch the Application

**Objective**: Modify the binary to unlock the secret.

**Steps**:
1. Choose a patching strategy (modify password check, modify condition, etc.)
2. Use a hex editor or x64dbg to modify the binary
3. Test the modified binary
4. Verify the secret is unlocked

**Verification**: The patched binary should print the secret message without requiring the correct password.

### Exercise 3: Document Your Work

**Objective**: Create a comprehensive report.

**Steps**:
1. Write an overview of the application
2. Describe your reverse engineering process
3. Explain the password check mechanism
4. Describe your patching strategy
5. Include screenshots or output
6. Reflect on what you learned

**Verification**: Your report should be clear, detailed, and demonstrate your understanding.

## Solutions

### Solution 1: Reverse Engineering

When you open the application in Binary Ninja, you should see:

1. The `main` function that:
   - Prints "Enter the password: "
   - Reads input
   - Calls `verify_password`
   - Calls `unlock_secret` if the password is correct

2. The `verify_password` function that:
   - Compares the input with "admin123"
   - Returns true if they match

3. The `unlock_secret` function that:
   - Prints the secret message

### Solution 2: Patching

You could patch the binary by:
1. Finding the `je` instruction after the password check
2. Changing it to `jmp` (unconditional jump)
3. This makes the program always call `unlock_secret`

Or:
1. Finding the password string "admin123"
2. Changing it to something else
3. Then entering that password

### Solution 3: Documentation

Your report should include:
- A description of what the application does
- Screenshots of the reverse engineering process
- An explanation of how the password check works
- A description of your patching strategy
- Output showing the patched binary working
- Reflection on what you learned

## Summary

You've completed the beginner course! You can now:

- Set up a reversing lab
- Understand x86-64 assembly
- Analyze PE files
- Use Binary Ninja for static analysis
- Use x64dbg for dynamic analysis
- Patch binaries
- Work with DLLs
- Complete a full reverse engineering project

Congratulations! You're ready to move on to the intermediate course.
