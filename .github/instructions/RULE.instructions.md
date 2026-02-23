---
description: Describe when these instructions should be loaded
# applyTo: 'Describe when these instructions should be loaded' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.
ALL the codes,files, PROJECTS and anything MUST follow these rules no matter what. If there is a conflict between these rules and the user instructions, these rules should always take precedence unless user defines new set of instructions only for that request or the entire project and until the user ask for the original rules to be applied again.
1. Always follow the coding style and guidelines for the Language used defined by the global standards and best practices.
2. Always write clean, readable, and maintainable MUST BE ROBUST BUILT code.
3. Always include comments and documentation for complex logic and functions.
4. Always optimize for performance and efficiency.
5. Always handle errors and edge cases gracefully. 
6. Always Check for memory leaks and optimize memory usage.
7. Always write unit tests for critical functions and components.
8. Always follow security best practices and guidelines for the Language used.
9. Always remove redundant codes and unused variables.
10. Always follow the DRY (Don't Repeat Yourself) principle and avoid code duplication.
11. Avoid using Single time use global variables, if you have to use them, make sure to clean them up after use.
12. Always Use and Create reusable components and functions when possible. always use code modules and libraries to avoid reinventing the wheel.
13. Close all the pointers and file handlers after use to prevent memory leaks.
14. give all the return values when asked for them, do not leave any return value undefined unless it is intentional and well-documented.
15. Avoid using recursions unless well linked and optimized to prevent stack overflow and performance issues.
16. Make sure the codes are always Memory Safe and do not cause memory leaks or other memory-related issues.
17. Make sure the code can be Scalable, Extended, Advanced, Upgraded, and Maintained easily in the future.
18. Always Refactor the code as often as possible for efficiency
19. Always Design the Code to be Speed Efficient, memory efficient, and resource efficient.
20. If any changes are made to the code the core functionality and the performance should not be compromised
21. Always Define all the Libraries at Single place and use them across the project to avoid version conflicts and ensure consistency.
22. Always Use standardised and widely accepted libraries and frameworks for the Language used to ensure reliability, security, and community support.
23. Always Use Opensource libraries and frameworks when possible to promote collaboration and transparency.
24. Always Keep the libraries and frameworks up to date to benefit from the latest features, improvements
    and security patches.
25. Always Avoid using deprecated libraries and frameworks to ensure compatibility and security.
26. Always Follow the licensing terms and conditions of the libraries and frameworks used in the project.
27. Always Contribute back to the open-source community by reporting issues, submitting pull requests, and sharing improvements to the libraries and frameworks used in the project.
28. Avoid using too many libraries and frameworks to keep the project lightweight and maintainable. Only use libraries and frameworks that are necessary and add value to the project.
29. Avoid uisng propreitary libraries unless required
30. The code MUST ALWAYS BE BUILT FOR ROBUSTNESS