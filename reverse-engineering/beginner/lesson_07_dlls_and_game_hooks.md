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

**DLL injection** is the technique of forcing a running process to load your custom DLL, giving you the ability to execute code within that process's address space. This is fundamental to game hacking, process monitoring, and many other reverse engineering tasks. Once your DLL is loaded in the target process, it has full access to the process's memory and can hook functions, modify data, or implement entirely new functionality.

### Method 1: CreateRemoteThread

The CreateRemoteThread injection method is one of the most common and well-documented DLL injection techniques. It works by creating a new thread in the target process that calls `LoadLibraryA` to load your DLL. Here's how the process works in detail.

First, you need to open the target process with `OpenProcess`. This Windows API function requires the process ID (which you can obtain from Task Manager or programmatically) and returns a handle to the process. You need to request sufficient access rights—specifically, `PROCESS_CREATE_THREAD`, `PROCESS_VM_OPERATION`, `PROCESS_VM_WRITE`, and `PROCESS_VM_READ`. These permissions allow you to create threads, allocate memory, and write to the process's memory.

Next, allocate memory in the target process using `VirtualAllocEx`. This function allocates a region of memory within the target process's address space. You need enough space to store the full path to your DLL (typically a few hundred bytes is sufficient). The allocated memory should have read/write permissions.

Once you have allocated memory, write the DLL path to that memory using `WriteProcessMemory`. This function copies data from your process into the target process. You're writing the full path to your DLL (as a null-terminated string) into the memory you just allocated. For example, you might write "C:\\Users\\YourName\\my_hook.dll" into the target process's memory.

Now comes the clever part: create a remote thread in the target process using `CreateRemoteThread`. The thread's start address should be the address of `LoadLibraryA` (which you can obtain using `GetProcAddress(GetModuleHandle("kernel32.dll"), "LoadLibraryA")`), and the thread parameter should be the address of the DLL path you wrote to the target process's memory. When this thread starts, it effectively calls `LoadLibraryA("C:\\Users\\YourName\\my_hook.dll")` within the target process, causing the target process to load your DLL.

Finally, the target process loads your DLL, and your `DllMain` function is called with `DLL_PROCESS_ATTACH`. At this point, your code is running inside the target process, and you can begin hooking functions or modifying behavior.

### Method 2: SetWindowsHookEx

The `SetWindowsHookEx` injection method is a more legitimate technique that leverages Windows's built-in hooking mechanism. Unlike CreateRemoteThread, which is often flagged by anti-cheat systems, SetWindowsHookEx is a documented Windows API designed for monitoring system events. However, it has a useful side effect: Windows automatically injects your DLL into any process that triggers your hook.

To use this method, you first create a DLL that exports a hook function. This function must match the signature required by the type of hook you're installing (for example, a keyboard hook, mouse hook, or window procedure hook). The hook function is called by Windows whenever the specified event occurs—for instance, whenever a key is pressed if you're using a keyboard hook.

Next, you call `SetWindowsHookEx` from your injector program, specifying the type of hook (like `WH_KEYBOARD` for keyboard events), the address of your hook function, the handle to your DLL, and the thread ID to hook (or 0 to hook all threads). When you install a global hook (thread ID = 0), Windows automatically injects your DLL into every process that receives the hooked events.

The system handles the injection automatically. When a process receives an event that triggers your hook (like a keystroke), Windows loads your DLL into that process and calls your hook function. This gives you code execution within the target process, and from there you can install additional hooks or modify behavior. This method is stealthier than CreateRemoteThread but is limited to processes that receive the events you're hooking.

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
