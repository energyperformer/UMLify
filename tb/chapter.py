import os
import fitz  # PyMuPDF
import re

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    # Replace common problematic characters with an underscore or remove them
    name = re.sub(r'[<>:"/\\|?*]', '_', name) # Replace with underscore
    name = re.sub(r'[\x00-\x1F]', '', name)    # Remove control characters
    
    # Windows reserved names check (case-insensitive for base name)
    # Create the base_name without the potential .txt yet for this check
    temp_base_name = name
    reserved_names_check = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    if temp_base_name.upper() in reserved_names_check:
        name += "_reserved"
        
    # Ensure filename is not empty or just dots
    if not name or name.strip() == "" or name.strip() == ".":
        name = "untitled_chapter"
        
    # Limit length to avoid issues (e.g., 200 chars for base name, then add .txt)
    name = name[:200]
    return name + ".txt"

def extract_chapter_titles_from_pdf(pdf_filename="Guide.pdf"):
    """
    Attempts to extract chapter titles from the PDF's Table of Contents.
    Assumes top-level entries (level 1) in ToC are main chapters.
    """
    extracted_titles = []
    print(f"Attempting to process PDF for ToC extraction: {pdf_filename}")
    try:
        if not os.path.exists(pdf_filename):
            print(f"Error: PDF file '{pdf_filename}' not found in '{os.getcwd()}'.")
            return []

        doc = fitz.open(pdf_filename)
        toc = doc.get_toc(simple=False)
        doc.close()

        if not toc:
            print(f"Warning: No Table of Contents (ToC) found in '{pdf_filename}'.")
            return []

        for entry in toc:
            if len(entry) >= 2 and entry[0] == 1:
                extracted_titles.append(entry[1])
        
        if extracted_titles:
            print(f"Successfully extracted {len(extracted_titles)} potential chapter titles from ToC of '{pdf_filename}'.")
        else:
            print(f"Warning: No level 1 ToC entries found in '{pdf_filename}'.")
        return extracted_titles
    except Exception as e:
        print(f"Error processing PDF '{pdf_filename}' for ToC: {e}")
        return []

def create_chapter_headings_file(output_headings_file="chapter_headings.txt"):
    """
    Creates a text file with chapter headings, primarily for review or as a basis.
    It first attempts to extract them from Guide.pdf's Table of Contents.
    If that fails, it falls back to a list of 27 placeholder titles.
    """
    pdf_titles = extract_chapter_titles_from_pdf("Guide.pdf")
    final_chapter_titles = []
    manual_fallback_needed = False

    if pdf_titles:
        final_chapter_titles = pdf_titles
        # print(f"Using {len(pdf_titles)} titles extracted from 'Guide.pdf' for {output_headings_file}.")
    else:
        # print(f"Automatic extraction for {output_headings_file} failed. Falling back to placeholders.")
        manual_fallback_needed = True
        final_chapter_titles = [f"Chapter {i+1}: Placeholder Title - Update Manually" for i in range(27)]
    
    try:
        with open(output_headings_file, "w", encoding="utf-8") as f:
            for title in final_chapter_titles:
                f.write(f"{title}\n")
        # print(f"Successfully created '{output_headings_file}'.")
        # if manual_fallback_needed: print(f"IMPORTANT: '{output_headings_file}' contains placeholders.")
    except IOError as e:
        print(f"Error writing to file '{output_headings_file}': {e}")

def extract_and_save_all_chapters_content(pdf_filename="Guide.pdf", chapter_headings_source="chapter_headings.txt"):
    """Reads chapter titles, finds them in PDF ToC, extracts text, and saves to individual files."""
    print(f"Starting chapter content extraction from '{pdf_filename}' based on '{chapter_headings_source}'.")

    if not os.path.exists(chapter_headings_source):
        print(f"Error: Chapter headings file '{chapter_headings_source}' not found. Please create it first.")
        print(f"You might need to run this script once in its original mode or ensure the file exists.")
        # Optionally, call create_chapter_headings_file here if desired as a fallback
        # print(f"Attempting to generate '{chapter_headings_source}' now...")
        # create_chapter_headings_file(chapter_headings_source)
        # if not os.path.exists(chapter_headings_source):
        #     print(f"Failed to create '{chapter_headings_source}'. Aborting.")
        #     return
        # print(f"'{chapter_headings_source}' generated. Please review it and re-run if necessary.")
        return

    source_titles = []
    with open(chapter_headings_source, 'r', encoding='utf-8') as f:
        source_titles = [line.strip() for line in f if line.strip()]

    if not source_titles:
        print(f"Error: '{chapter_headings_source}' is empty. No titles to process.")
        return

    # Assume the last title is "Contents" and remove it if present
    processed_titles_for_extraction = list(source_titles) # Make a copy
    if processed_titles_for_extraction and processed_titles_for_extraction[-1].strip().lower() == "contents":
        print(f"Removing last title 'Contents' from the list for chapter extraction.")
        processed_titles_for_extraction.pop()
    
    print(f"Will attempt to extract content for {len(processed_titles_for_extraction)} chapters.")

    if not os.path.exists(pdf_filename):
        print(f"Error: PDF file '{pdf_filename}' not found. Cannot extract content.")
        return

    doc = None
    try:
        doc = fitz.open(pdf_filename)
        full_toc = doc.get_toc(simple=False) # Get [level, title, page_num, dest_dict]

        if not full_toc:
            print(f"Warning: No Table of Contents found in '{pdf_filename}'. Cannot map chapters to pages.")
            return

        chapter_details_from_pdf = []
        for title_to_find in processed_titles_for_extraction:
            found_in_toc = False
            for toc_entry in full_toc:
                toc_level, toc_title_pdf, toc_page_num_1idx = toc_entry[0], toc_entry[1], toc_entry[2]
                # Match based on level 1 and exact title match (after stripping)
                if toc_level == 1 and toc_title_pdf.strip() == title_to_find.strip():
                    chapter_details_from_pdf.append({
                        'title_from_file': title_to_find,
                        'toc_title': toc_title_pdf,
                        'start_page_1idx': toc_page_num_1idx
                    })
                    found_in_toc = True
                    break # Found the title, move to next title_to_find
            if not found_in_toc:
                print(f"Warning: Could not find a matching level 1 ToC entry for chapter: '{title_to_find}'. It will be skipped.")
        
        if not chapter_details_from_pdf:
            print("No chapter titles from the headings file could be matched in the PDF's Table of Contents. Aborting extraction.")
            return

        # Sort the found chapters by their start page number from ToC
        chapter_details_from_pdf.sort(key=lambda x: x['start_page_1idx'])
        print(f"Successfully matched {len(chapter_details_from_pdf)} chapters from headings file to PDF ToC and sorted them by page.")

        for i, chap_info in enumerate(chapter_details_from_pdf):
            title = chap_info['title_from_file'] # Use the title from the headings file for consistency
            start_page_0idx = chap_info['start_page_1idx'] - 1
            
            end_page_0idx_exclusive = 0
            if i == len(chapter_details_from_pdf) - 1: # Last chapter in our matched list
                end_page_0idx_exclusive = doc.page_count
            else:
                next_chap_start_page_1idx = chapter_details_from_pdf[i+1]['start_page_1idx']
                end_page_0idx_exclusive = next_chap_start_page_1idx - 1
            
            print(f"Processing chapter: '{title}' (Pages {start_page_0idx + 1} to {end_page_0idx_exclusive})")

            chapter_text_content = ""
            if start_page_0idx >= end_page_0idx_exclusive:
                print(f"  Warning: Chapter '{title}' has an invalid page range (start page {start_page_0idx + 1} >= end page {end_page_0idx_exclusive}). Content might be missing or on a single page not covered if start=end. Skipping text extraction for this chapter.")
            else:
                for p_num in range(start_page_0idx, end_page_0idx_exclusive):
                    if p_num >= doc.page_count:
                        print(f"  Warning: Page number {p_num + 1} is out of bounds for PDF ({doc.page_count} pages). Stopping extraction for '{title}'.")
                        break
                    page = doc.load_page(p_num)
                    chapter_text_content += page.get_text("text")
                    chapter_text_content += "\n\n" # Add a separator between text from different pages
            
            output_txt_filename = sanitize_filename(title)
            try:
                with open(output_txt_filename, "w", encoding="utf-8") as f_out:
                    f_out.write(chapter_text_content.strip()) # Strip trailing newlines from the last page
                print(f"  Successfully extracted and saved to '{output_txt_filename}' ({len(chapter_text_content.strip())} characters).")
            except IOError as e_write:
                print(f"  Error writing chapter '{title}' to file '{output_txt_filename}': {e_write}")

    except Exception as e_main:
        print(f"An error occurred during chapter content extraction: {e_main}")
    finally:
        if doc:
            doc.close()
            print("Closed PDF document.")
    print("Chapter content extraction process finished.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current working directory set to: {os.getcwd()}")
    
    # The new primary function for this script
    extract_and_save_all_chapters_content()
    
    # To generate the chapter_headings.txt initially (if needed), one could uncomment below
    # print("\n--- Running initial heading file creation (if needed) ---")
    # create_chapter_headings_file("chapter_headings.txt")
