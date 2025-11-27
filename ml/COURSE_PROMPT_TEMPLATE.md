# Course Generation Prompt Template

This file contains a reusable meta-prompt for generating a **three-level course series** (Beginner, Intermediate, Advanced) on any software-engineering-related topic.

To use it, replace placeholders like `{TOPIC}` and `{PRIMARY_LANGS}` with your specific choices, then paste the prompt into the model.

---

```text
You are an expert software engineering educator and curriculum designer.
Your task is to design a **three-part course series** on **{TOPIC}** for professional software engineers and aspiring developers.

The three courses must be:

1. **Beginner {TOPIC} course**
   - Audience: Has **no prior experience** with {TOPIC} and minimal to no background in this sub-area. May know general programming (e.g. basic Python/JavaScript), but is new to this specific topic.
   - Goal: By the end, they should understand core concepts, vocabulary, and be able to complete small, guided projects with support.

2. **Intermediate {TOPIC} course**
   - Audience: Has completed the beginner course or equivalent. Knows the basics and has done small projects, but lacks depth, patterns, and confidence.
   - Goal: By the end, they should be able to design and implement **non-trivial, real-world projects** in {TOPIC} mostly independently, using best practices.

3. **Advanced {TOPIC} course**
   - Audience: Has completed the intermediate course or equivalent. Comfortable building production-style systems and wants to reach **expert level**: deep understanding, performance, reliability, architecture, and trade-offs.
   - Goal: By the end, they should be able to **design, critique, and optimize** complex systems in {TOPIC}, mentor others, and make informed architectural decisions.

Global constraints:

- **Target audience:** Software engineers, data engineers, ML engineers, or CS students. Assume they can code, but adjust depth according to level (beginner vs advanced in this topic).
- **Teaching style:**
  - Build concepts progressively from fundamentals.
  - Explain all new terminology clearly when first introduced.
  - Prefer **clear paragraphs** over dense bullet lists; use bullets only for short enumerations.
  - Always explain the **"why"** behind concepts and design choices: trade-offs, alternatives, and when/when not to use something.
  - When math or formalism is needed, walk through it **step by step**, with intuition and simple examples.
- **Practical focus:** These courses are for people who want to *do* things. For every theoretical idea, connect it to a practical use case and, when possible, a code example.
- **Technologies / languages:** Focus examples and exercises on {PRIMARY_LANGS} and the most common stacks for {TOPIC}. If a choice is ambiguous, pick pragmatic, widely used tools.

Output format: produce Markdown with this structure.

1. **High-level overview of the 3-course series**
   - Short description of each course (Beginner, Intermediate, Advanced) and how they fit together.
   - Suggested total duration (e.g. hours or weeks) per course.

2. **Beginner Course: {TOPIC} Basics**
   - A short paragraph describing the target learner and goals.
   - A numbered list of **8–12 lessons**, each with:
     - `Lesson N: Title`
     - 1–2 sentences on what the lesson covers.
   - Then, for **each lesson**, a dedicated subsection with:
     - A short "why this matters" intro.
     - Detailed explanations of the core concepts in clear paragraphs.
     - At least one **concrete example**, ideally including short code snippets relevant to {TOPIC}.
     - **Exercises section** with **3–8 exercises** that require hands-on work (coding, designing APIs, writing tests, debugging code, etc.).
     - **Solutions** immediately after the exercises, under a clearly marked "Solutions" subheading, with fully worked answers (not just one-word answers).

3. **Intermediate Course: {TOPIC} in Practice**
   - Short paragraph on audience, prerequisites (assumes Beginner course), and goals.
   - Numbered list of **8–12 lessons**, each with a title and brief description.
   - For each lesson:
     - Deeper, more detailed explanations of concepts and patterns than in the beginner course.
     - Emphasis on **design decisions**, best practices, and common pitfalls.
     - Realistic code examples (e.g. small services, modules, scripts, pipelines, or components) in {PRIMARY_LANGS}.
     - **Exercises** (3–8 per lesson) focused on extending code, refactoring, performance improvements, adding tests, debugging, and integrating with other components.
     - **Solutions** for every exercise, with full code and reasoning.
   - Include at least **one capstone project** for the intermediate course:
     - A non-trivial, real-world style project description.
     - Step-by-step guidance / milestones.
     - A high-level solution outline (architecture, main components, hints for implementation).

4. **Advanced Course: {TOPIC} Mastery**
   - Short paragraph on audience and advanced goals (expertise, architecture, scalability, reliability, performance, etc.).
   - Numbered list of **8–12 advanced lessons**, each with a title and 1–3 sentences, focusing on topics like architecture patterns, trade-offs, scalability, reliability, observability, performance tuning, security, etc. (tailor to {TOPIC}).
   - For each advanced lesson:
     - In-depth conceptual explanation with focus on real-world constraints and trade-offs.
     - Case studies or mini-postmortems where relevant.
     - Non-trivial code examples, design diagrams described in text, or pseudo-code where appropriate.
     - **Exercises** (3–8 per lesson) that might include: explaining design choices, evaluating trade-offs, optimizing existing solutions, designing robust APIs, adding observability, assessing security implications, etc.
     - **Solutions** for every exercise, with detailed justifications, not just final code.
   - At least one **major capstone project**:
     - A substantial real-world style project that integrates topics across all three courses.
     - Clear requirements, constraints, and evaluation criteria.
     - A detailed, high-level solution and architecture, with notes on extensions for even more advanced exploration.

Style and quality requirements:

- **No prior knowledge of {TOPIC} is assumed** in the beginner course; explain like you would to a smart software engineer encountering it for the first time.
- Use **plain language** and avoid unexplained jargon; when jargon is necessary, define it immediately and clearly.
- Prefer **narrative explanations in paragraphs**; use bullet lists mainly for checklists, learning objectives, or quick summaries.
- For all code examples and solutions, ensure they are correct in principle, idiomatic for the chosen language/stack, and minimal yet illustrative.
- Include **debugging and testing tasks** among the exercises, not just "write new code from scratch".
- Ensure the progression across the three courses feels coherent: advanced concepts should build on or clearly relate back to earlier material.

Final instruction:

Using all the constraints and structure above, design the full **three-course series on {TOPIC}** in Markdown, including:
- Course overviews,
- Lesson lists,
- Full lesson content (explanations + code examples),
- Exercises **with solutions** for every lesson,
- And at least one capstone project each for the intermediate and advanced courses.

Do **not** skip the solutions.
```

