import os

# Get all files in the current directory
files = os.listdir('.')

# Open the toc.md file (create it if it doesn't exist)
with open('toc.md', 'w') as toc:
    # Loop over all files
    for filename in files:
        # Check if the file is a markdown file
        if filename.endswith('.md') and filename != 'toc.md':
            # Open the markdown file
            with open(filename, 'r') as md_file:
                # Read the first line
                first_line = md_file.readline().strip()
                # Check if the first line is a title (starts with '#')
                if first_line.startswith('#'):
                    # Write the title and the file name to the toc.md file
                    toc.write(f'[{first_line[1:].strip()}]({filename})\n')
