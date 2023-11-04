def read_and_compute_avg(file_name):
    # Initialize sum and count
    total_time = 0.0
    count = 0
    
    # Open and read file
    with open(file_name, 'r') as file:
        for line in file:
            # Split the line and extract the time taken in seconds
            time_taken = float(line.split(":")[1].split("seconds")[0].strip())
            total_time += time_taken
            count += 1

    # Calculate average
    average_time_seconds = round(total_time / count,0)
    average_time_hours = round(average_time_seconds / 3600, 2)
    
    # Print results
    print(f"Average time in seconds: {average_time_seconds}")
    print(f"Average time in hours: {average_time_hours}")

# File namem
model = "16B"
pass_at = 2
test_lines = 1
file_name = f"batch_{model}_p{pass_at}_t{test_lines}.txt"
read_and_compute_avg(file_name)
