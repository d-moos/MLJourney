# Lesson 7: DLLs and Game Hooking – Injecting Code into Running Processes

## Overview

A **DLL** (Dynamic Link Library) is a Windows library that can be loaded by a process at runtime. **DLL injection** is a technique where you create a DLL and inject it into a running process. This allows you to:
- Hook functions in the target process
- Modify behavior without patching the binary
- Add new functionality
- Monitor function calls

This is commonly used in game modding, where you inject a DLL into a game to add features or change behavior.

## What You'll Learn

By the end of this lesson, you will understand:

- **How DLLs work** and how they're loaded
- **How to create a simple DLL**
- **How to inject a DLL** into a running process
- **How to hook functions** from within a DLL
- **How to communicate** between your DLL and the host process

## Prerequisites

Before starting this lesson, you should:

- Have completed Lessons 1-6
- Be comfortable with Rust or C/C++
- Understand function hooking concepts

## What is a DLL?

A DLL is a library containing code and data that can be loaded by multiple processes. Unlike an executable (.exe), a DLL doesn't run on its own—it's loaded by another process.

### DLL Entry Point

Every DLL has an entry point function called `DllMain`:

```rust
#[no_mangle]
pub extern "system" fn DllMain(
    _module: *mut std::ffi::c_void,
    call_reason: u32,
    _reserved: *mut std::ffi::c_void,
) -> i32 {
    match call_reason {
        1 => {  // DLL_PROCESS_ATTACH
            // DLL is being loaded
            println!("DLL loaded!");
        }
        0 => {  // DLL_PROCESS_DETACH
            // DLL is being unloaded
            println!("DLL unloaded!");
        }
        _ => {}
    }
    1  // Return TRUE
}
```

When the DLL is loaded, `DllMain` is called with `call_reason = 1` (DLL_PROCESS_ATTACH). When it's unloaded, `call_reason = 0` (DLL_PROCESS_DETACH).

## Creating a Simple DLL

Here's a minimal Rust DLL:

```rust
#[no_mangle]
pub extern "system" fn DllMain(
    _module: *mut std::ffi::c_void,
    call_reason: u32,
    _reserved: *mut std::ffi::c_void,
) -> i32 {
    if call_reason == 1 {
        // DLL loaded
        unsafe {
            // Your code here
        }
    }
    1
}
```

To compile as a DLL:
```
rustc --crate-type cdylib my_dll.rs
```

## DLL Injection

**DLL injection** is the process of loading your DLL into a running process. There are several techniques:

### Method 1: CreateRemoteThread

1. Open the target process with `OpenProcess`
2. Allocate memory in the target process with `VirtualAllocEx`
3. Write the DLL path to that memory with `WriteProcessMemory`
4. Create a remote thread that calls `LoadLibraryA` with the DLL path
5. The target process loads your DLL

### Method 2: SetWindowsHookEx

1. Create a DLL with a hook function
2. Use `SetWindowsHookEx` to install the hook
3. The system automatically injects your DLL into processes that trigger the hook

## Hooking Functions from a DLL

Once your DLL is loaded, you can hook functions:

```rust
// Original function pointer
type GetStdHandleFunc = extern "system" fn(u32) -> *mut std::ffi::c_void;
static mut ORIGINAL_GET_STD_HANDLE: Option<GetStdHandleFunc> = None;

// Hook function
extern "system" fn hooked_get_std_handle(handle: u32) -> *mut std::ffi::c_void {
    println!("GetStdHandle called with: {}", handle);
    
    // Call original function
    unsafe {
        ORIGINAL_GET_STD_HANDLE.unwrap()(handle)
    }
}

// In DllMain:
// 1. Get address of original GetStdHandle
// 2. Replace it with hooked_get_std_handle
// 3. Save original address
```

## Exercises

### Exercise 1: Create a Simple DLL

**Objective**: Learn to create a DLL.

**Steps**:
1. Create a Rust project: `cargo new --lib my_dll`
2. In `Cargo.toml`, set `crate-type = ["cdylib"]`
3. Write a `DllMain` function that prints a message
4. Compile: `cargo build --release`
5. The DLL is at `target/release/my_dll.dll`

**Verification**: You should have a working DLL file.

### Exercise 2: Inject a DLL into a Process

**Objective**: Learn to inject a DLL.

**Steps**:
1. Create a simple target process (e.g., a Rust program that runs for 10 seconds)
2. Create an injector program that:
   - Finds the target process
   - Opens it with `OpenProcess`
   - Allocates memory with `VirtualAllocEx`
   - Writes the DLL path with `WriteProcessMemory`
   - Creates a remote thread with `CreateRemoteThread`
3. Run the target process
4. Run the injector
5. Verify the DLL was loaded (check for output or side effects)

**Verification**: The DLL should be loaded into the target process.

### Exercise 3: Hook a Function from a DLL

**Objective**: Learn to hook functions.

**Steps**:
1. Create a DLL that hooks `GetStdHandle`
2. The hook should print a message before calling the original function
3. Inject the DLL into a process that calls `GetStdHandle`
4. Verify the hook is called

**Verification**: You should see the hook's message printed.

## Solutions

### Solution 1: Create a Simple DLL

A simple DLL in Rust:

```rust
#[no_mangle]
pub extern "system" fn DllMain(
    _module: *mut std::ffi::c_void,
    call_reason: u32,
    _reserved: *mut std::ffi::c_void,
) -> i32 {
    if call_reason == 1 {
        println!("DLL loaded!");
    }
    1
}
```

### Solution 2: Inject a DLL

An injector program would:
1. Find the target process by name
2. Open it with `OpenProcess`
3. Allocate memory with `VirtualAllocEx`
4. Write the DLL path with `WriteProcessMemory`
5. Create a remote thread with `CreateRemoteThread` that calls `LoadLibraryA`

### Solution 3: Hook a Function

A hook function would:
1. Save the original function address
2. Replace it with the hook function
3. In the hook, call the original function and return its result

## Summary

You now know how to work with DLLs and inject code into processes. You can:

- Create DLLs
- Inject DLLs into running processes
- Hook functions from within a DLL
- Modify process behavior without patching

In the next lesson, you'll complete a capstone project that brings together everything you've learned.
