from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # Initialize a list to store the groups
    groups = []
    # Initialize a variable to store the current group
    current_group = ''
    # Initialize a variable to store the number of open braces
    open_braces = 0
    # Loop through each character in the input string
    for char in paren_string:
        # If the character is an open brace, increment the number of open braces
        if char == '(':
            open_braces += 1
        # If the character is a closed brace, decrement the number of open braces
        elif char == ')':
            open_braces -= 1
        # If the character is a space, ignore it
        elif char =='':
            continue
        # If the number of open braces is 0, that means we have reached the end of a group
        if open_braces == 0:
            # Append the current group to the list of groups
            groups.append(current_group)
            # Reset the current group and the number of open braces
            current_group = ''
            open_braces = 0
        # Otherwise, add the character to the current group
        else:
            current_group += char
    # Return the list of groups
    return groups

print(separate_paren_groups('( ) (( )) (( )( ))'))