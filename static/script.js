//runs when new img is selected
function handleImagePreview(event) {
    var file = event.target.files[0];
    var reader = new FileReader();

    reader.onload = function(e) {
        var imagePreview = document.getElementById("image-preview");
        imagePreview.src = e.target.result;
        var imagePreviewContainer = document.getElementById("image-preview-container");
        imagePreviewContainer.classList.add("show");
        document.getElementById("result-text").innerHTML = "";
    }

    reader.readAsDataURL(file);
}
