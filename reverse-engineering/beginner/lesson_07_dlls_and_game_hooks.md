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

Creating a DLL in Rust requires understanding the DLL entry point and proper configuration. Here's a complete working example:

First, create a new Rust library project with `cargo new --lib my_dll`. Then, modify your `Cargo.toml` to specify that you want to build a C-compatible dynamic library:

```toml
[lib]
crate-type = ["cdylib"]
```

The `cdylib` crate type tells Rust to create a C-compatible dynamic library (DLL on Windows). This is different from `dylib`, which creates a Rust-specific dynamic library.

Now, in your `lib.rs`, implement the `DllMain` function:

```rust
#[no_mangle]
pub extern "system" fn DllMain(
    _module: *mut std::ffi::c_void,
    call_reason: u32,
    _reserved: *mut std::ffi::c_void,
) -> i32 {
    if call_reason == 1 {  // DLL_PROCESS_ATTACH
        println!("DLL loaded!");
    } else if call_reason == 0 {  // DLL_PROCESS_DETACH
        println!("DLL unloaded!");
    }
    1  // Return TRUE
}
```

The `#[no_mangle]` attribute prevents Rust from mangling the function name, ensuring it's exported as `DllMain` exactly. The `extern "system"` specifies the Windows calling convention. The function returns 1 (TRUE) to indicate successful initialization.

When you compile with `cargo build --release`, you'll find your DLL at `target/release/my_dll.dll`. You can verify it's a valid DLL by loading it in a PE viewer or by attempting to inject it into a process.

### Solution 2: Inject a DLL

DLL injection requires careful use of Windows APIs to manipulate another process's memory. Here's a detailed walkthrough of implementing a CreateRemoteThread injector:

First, you need to find the target process. You can do this by enumerating processes with `CreateToolhelp32Snapshot` and `Process32Next`, searching for a process by name. Once you have the process ID, open the process with `OpenProcess`, requesting the necessary permissions:

```rust
let process_handle = OpenProcess(
    PROCESS_CREATE_THREAD | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
    false,
    target_pid
);
```

Next, allocate memory in the target process to store the DLL path. The path must be a full absolute path (like "C:\\Users\\YourName\\my_dll.dll"):

```rust
let dll_path = "C:\\path\\to\\my_dll.dll\0";  // Null-terminated
let path_size = dll_path.len();
let remote_memory = VirtualAllocEx(
    process_handle,
    null_mut(),
    path_size,
    MEM_COMMIT | MEM_RESERVE,
    PAGE_READWRITE
);
```

Write the DLL path to the allocated memory:

```rust
WriteProcessMemory(
    process_handle,
    remote_memory,
    dll_path.as_ptr() as *const _,
    path_size,
    null_mut()
);
```

Now comes the clever part: get the address of `LoadLibraryA` from `kernel32.dll`. This address is the same in all processes (due to ASLR being consistent across processes), so you can get it from your own process:

```rust
let kernel32 = GetModuleHandleA("kernel32.dll\0".as_ptr() as *const i8);
let load_library_addr = GetProcAddress(kernel32, "LoadLibraryA\0".as_ptr() as *const i8);
```

Finally, create a remote thread that calls `LoadLibraryA` with the DLL path as the argument:

```rust
let thread_handle = CreateRemoteThread(
    process_handle,
    null_mut(),
    0,
    std::mem::transmute(load_library_addr),  // Thread start address
    remote_memory,  // Thread parameter (DLL path)
    0,
    null_mut()
);
```

When this thread starts in the target process, it executes `LoadLibraryA("C:\\path\\to\\my_dll.dll")`, causing the target process to load your DLL. Your `DllMain` function will be called with `DLL_PROCESS_ATTACH`, and you'll see "DLL loaded!" printed (assuming the target process has a console or you're using a message box instead of `println!`).

### Solution 3: Hook a Function

Hooking a function from within a DLL requires saving the original function address, replacing it with your hook, and ensuring your hook can call the original. Here's a detailed implementation for hooking `GetStdHandle`:

First, define a type for the original function signature:

```rust
type GetStdHandleFunc = extern "system" fn(u32) -> *mut std::ffi::c_void;
static mut ORIGINAL_GET_STD_HANDLE: Option<GetStdHandleFunc> = None;
```

Implement your hook function:

```rust
extern "system" fn hooked_get_std_handle(std_handle: u32) -> *mut std::ffi::c_void {
    println!("GetStdHandle called with: {}", std_handle);

    // Call the original function
    unsafe {
        if let Some(original) = ORIGINAL_GET_STD_HANDLE {
            original(std_handle)
        } else {
            null_mut()
        }
    }
}
```

In your `DllMain`, install the hook when the DLL is loaded:

```rust
if call_reason == 1 {  // DLL_PROCESS_ATTACH
    unsafe {
        // Get the address of the original GetStdHandle
        let kernel32 = GetModuleHandleA("kernel32.dll\0".as_ptr() as *const i8);
        let original_addr = GetProcAddress(kernel32, "GetStdHandle\0".as_ptr() as *const i8);
        ORIGINAL_GET_STD_HANDLE = Some(std::mem::transmute(original_addr));

        // Install the hook (this is simplified - real hooking requires more work)
        // You would typically use a hooking library like MinHook or implement
        // inline hooking by overwriting the first bytes of GetStdHandle with a jump
    }
}
```

For a complete implementation, you'd typically use a hooking library like MinHook (available as a Rust crate) which handles the complex details of inline hooking: saving the original bytes, writing a jump instruction, creating a trampoline, and handling thread safety. The basic principle is that you overwrite the first 5+ bytes of `GetStdHandle` with a `jmp` instruction to your hook function, save those original bytes in a trampoline, and have your hook call the trampoline to execute the original function.

## Summary

You now know how to work with DLLs and inject code into processes. You can:

- Create DLLs
- Inject DLLs into running processes
- Hook functions from within a DLL
- Modify process behavior without patching

In the next lesson, you'll complete a capstone project that brings together everything you've learned.
