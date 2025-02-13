from flask import Flask, request, jsonify
from deepface import DeepFace
import logging
from typing import List, Dict, Set
import os

class EnhancedFamilyGrouper:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.model_name = "VGG-Face"
        self.face_cache = {}  # Cache face counts to avoid reprocessing
        logging.basicConfig(level=logging.INFO)
        
    def count_faces(self, image_path: str) -> int:
        if image_path in self.face_cache:
            return self.face_cache[image_path]
        
        try:
            faces = DeepFace.extract_faces(img_path=image_path)
            count = len(faces)
            self.face_cache[image_path] = count
            return count
        except Exception as e:
            logging.error(f"Error counting faces in {image_path}: {str(e)}")
            return 0
            
    def is_family_photo(self, photo_path: str) -> bool:
        return self.count_faces(photo_path) > 1
        
    def compare_faces(self, img1_path: str, img2_path: str) -> float:
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                distance_metric="cosine"
            )
            return result["distance"]
        except Exception as e:
            logging.error(f"Error comparing faces: {str(e)}")
            return float('inf')

    def find_matching_individuals(self, family_photo: str, all_photos: List[str]) -> Set[str]:
        """Find all individual photos that match faces in the family photo"""
        matching_photos = set()
        
        for photo in all_photos:
            if self.count_faces(photo) == 1:  # Only check individual photos
                if self.compare_faces(family_photo, photo) <= self.threshold:
                    matching_photos.add(photo)
                    logging.info(f"Found matching individual photo: {photo}")
                    
        return matching_photos

    def merge_groups(self, groups: Dict[int, List[str]], group_ids: List[int]) -> Dict[int, List[str]]:
        """Merge multiple groups into the first group"""
        if not group_ids:
            return groups
            
        target_group = group_ids[0]
        all_photos = set(groups[target_group])
        
        # Collect all photos from groups to be merged
        for group_id in group_ids[1:]:
            all_photos.update(groups[group_id])
            del groups[group_id]
            
        groups[target_group] = list(all_photos)
        return groups

    def group_photos(self, photos: List[str]) -> Dict[int, List[str]]:
        groups: Dict[int, List[str]] = {}
        current_group = 0
        
        # First pass: Group individual photos
        individual_photos = [p for p in photos if self.count_faces(p) == 1]
        logging.info(f"Processing individual photos first: {individual_photos}")
        
        for photo in individual_photos:
            matched_group = -1
            
            for group_id, group_photos in groups.items():
                for group_photo in group_photos:
                    if self.compare_faces(photo, group_photo) <= self.threshold:
                        matched_group = group_id
                        break
                if matched_group >= 0:
                    break
                    
            if matched_group >= 0:
                groups[matched_group].append(photo)
                logging.info(f"Added individual photo {photo} to Group {matched_group}")
            else:
                groups[current_group] = [photo]
                logging.info(f"Created new Group {current_group} for individual photo {photo}")
                current_group += 1
        
        # Second pass: Process family photos and merge groups
        family_photos = [p for p in photos if self.count_faces(p) > 1]
        logging.info(f"Processing family photos: {family_photos}")
        
        for family_photo in family_photos:
            # Find all individual photos that match people in this family photo
            matching_individuals = self.find_matching_individuals(family_photo, photos)
            
            # Find all groups containing these matching individuals
            groups_to_merge = []
            for group_id, group_photos in groups.items():
                if any(photo in matching_individuals for photo in group_photos):
                    groups_to_merge.append(group_id)
                    
            if groups_to_merge:
                # Add family photo to first group and merge all related groups
                if family_photo not in groups[groups_to_merge[0]]:
                    groups[groups_to_merge[0]].append(family_photo)
                groups = self.merge_groups(groups, groups_to_merge)
                logging.info(f"Merged groups {groups_to_merge} after processing family photo {family_photo}")
            else:
                # Create new group if no matches found
                groups[current_group] = [family_photo]
                logging.info(f"Created new Group {current_group} for family photo {family_photo}")
                current_group += 1
                
        return groups

# Flask Application Setup
app = Flask(__name__)
face_grouper = EnhancedFamilyGrouper()

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "deepface-grouping"
    }), 200

@app.route('/group', methods=['POST'])
def group_photos():
    """Main endpoint for photo grouping"""
    try:
        data = request.json
        photos = data.get('photos', [])
        
        if len(photos) < 1:
            return jsonify({"error": "At least one photo is required"}), 400
            
        # Validate all photos exist
        for photo in photos:
            if not os.path.exists(photo):
                return jsonify({"error": f"Photo not found: {photo}"}), 404
                
        # Group photos
        groups = face_grouper.group_photos(photos)
        
        # Add metadata about groups
        group_metadata = {}
        for group_id, group_photos in groups.items():
            group_metadata[group_id] = {
                "photos": group_photos,
                "family_photos": sum(1 for p in group_photos if face_grouper.is_family_photo(p)),
                "individual_photos": sum(1 for p in group_photos if not face_grouper.is_family_photo(p)),
                "total_photos": len(group_photos),
                "members": len([p for p in group_photos if not face_grouper.is_family_photo(p)])  # Count of individuals
            }
        
        return jsonify({
            "status": "success",
            "groups": group_metadata,
            "total_groups": len(groups),
            "total_photos": len(photos)
        })
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_photo():
    """Endpoint for analyzing a single photo"""
    try:
        data = request.json
        photo = data.get('photo')
        
        if not photo:
            return jsonify({"error": "Photo path is required"}), 400
            
        if not os.path.exists(photo):
            return jsonify({"error": "Photo not found"}), 404
            
        face_count = face_grouper.count_faces(photo)
        
        return jsonify({
            "status": "success",
            "photo": photo,
            "face_count": face_count,
            "is_family_photo": face_count > 1
        })
        
    except Exception as e:
        logging.error(f"Error analyzing photo: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)