# Import the required task function 'add' from the 'tasks' module
from tasks import simple_task

# Call the 'add' task function asynchronously and store the result
result = simple_task.delay()

# Check if the result is ready (completed)
print(result.ready())

# Print the status of the result
print(result.status)

# Print the actual result of the task (if it's completed)
print(result.result)

# Print the status of the result again (to demonstrate that it remains unchanged)
print(result.status)

# Print the result object itself (useful for debugging and introspection)
print(result.get())