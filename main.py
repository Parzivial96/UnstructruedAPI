from flask import Flask, request, jsonify
from unstructured.partition.pdf import partition_pdf

app = Flask(__name__)

def merge_bounding_boxes(points_list):
    min_x = min(point[0] for point in points_list[0])
    max_x = max(point[0] for point in points_list[0])
    min_y = min(point[1] for point in points_list[0])
    max_y = max(point[1] for point in points_list[0])

    for points in points_list[1:]:
        min_x = min(min_x, min(point[0] for point in points))
        max_x = max(max_x, max(point[0] for point in points))
        min_y = min(min_y, min(point[1] for point in points))
        max_y = max(max_y, max(point[1] for point in points))

    return ((min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y))

def process_pdf(file_stream, resource_id):
    print("Starting chunking...")
    file_stream.seek(0)
    try:
        chunks = partition_pdf(
            file=file_stream,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=1000
        )
    except Exception as e:
        print(f"Error during PDF partitioning: {e}")
        raise

    print("Chunking done.")

    data = []
    current_text_group = []
    current_bounding_boxes = []
    current_page_number = 1
    current_bounding_box_merged = []

    def make_current_page_bounding_box():
        current_bounding_box_merged.append(
            {
                current_page_number: merge_bounding_boxes(current_bounding_boxes)
            }
        )
        current_bounding_boxes.clear()

    def flush_text_group():
        if current_text_group:
            combined_content = " ".join(current_text_group).strip()
            if combined_content:
                data.append({
                    "id": resource_id + "_" + str(len(data)),
                    "content": combined_content,
                    "type": "text",
                    "bounding_box": current_bounding_box_merged.copy(),
                })
        current_text_group.clear()
        current_bounding_box_merged.clear()
    
    i=0
    for chunk in chunks:
        print(f"Chunk {i} is being processed.")

        orig_elements = chunk.metadata.orig_elements
        for orig_element in orig_elements:
            if "Image" in str(type(orig_element)):
                data.append({
                    "id": resource_id + "_" + str(len(data)),
                    "content": orig_element.metadata.image_base64,
                    "type": "image",
                    "mime_type": orig_element.metadata.image_mime_type,
                    "bounding_box": [{orig_element.metadata.page_number: orig_element.metadata.coordinates.points}]
                })
            elif "Table" in str(type(orig_element)):
                data.append({
                    "id": resource_id + "_" + str(len(data)),
                    "content": orig_element.metadata.image_base64,
                    "type": "table",
                    "mime_type": orig_element.metadata.image_mime_type,
                    "bounding_box": [{orig_element.metadata.page_number: orig_element.metadata.coordinates.points}]
                })
            else:
                if orig_element.metadata.page_number != current_page_number:
                    if len(current_bounding_boxes) > 0:
                        make_current_page_bounding_box()
                    current_page_number = orig_element.metadata.page_number

                current_text_group.append(orig_element.text)
                current_bounding_boxes.append(orig_element.metadata.coordinates.points)

        make_current_page_bounding_box()
        flush_text_group()
        i+=1

    print("Data processing done.")

    return data

@app.route('/process-pdf', methods=['POST'])
def process_pdf_endpoint():
    try:
        resource_id = request.form.get('resource_id')
        file = request.files.get('file')
        file.save("temp_file.pdf")

        if not resource_id or not file:
            return jsonify({"error": "Missing resource_id or file in request"}), 400
        
        result = process_pdf(file.stream, resource_id)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
