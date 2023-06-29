function drawTree(treeData) {
    var width = 800;
    var height = 600;

    var svg = d3.select("#tree-container")
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .append("g")
                .attr("transform", "translate(50, 50)");

    var treeLayout = d3.tree().size([width - 200, height - 200]);

    var root = d3.hierarchy(treeData);
    treeLayout(root);

    var nodes = root.descendants();
    var links = root.links();

    var link = svg.selectAll(".link")
                    .data(links)
                    .enter()
                    .append("path")
                    .attr("class", "link")
                    .attr("d", function(d) {
                        return "M" + d.source.x + "," + d.source.y
                                + "C" + d.source.x + "," + (d.source.y + d.target.y) / 2
                                + " " + d.target.x + "," + (d.source.y + d.target.y) / 2
                                + " " + d.target.x + "," + d.target.y;
                    });

    var node = svg.selectAll(".node")
                    .data(nodes)
                    .enter()
                    .append("g")
                    .attr("class", "node")
                    .attr("transform", function(d) {
                        return "translate(" + d.x + "," + d.y + ")";
                    });

    node.append("circle")
        .attr("r", 15)
        .style("fill", "#fff")
        .style("stroke", "#4287f5")
        .style("stroke-width", "2px");

    node.append("text")
        .attr("dy", "5px")
        .style("text-anchor", "middle")
        .text(function(d) {
            if (d.data.label) {
                return d.data.label;
            } else {
                return d.data.split_attribute;
            }
        });

    var label = svg.selectAll(".label")
                    .data(links)
                    .enter()
                    .append("text")
                    .attr("class", "label")
                    .attr("transform", function(d) {
                        var midX = (d.source.x + d.target.x) / 2;
                        var midY = (d.source.y + d.target.y) / 2;
                        return "translate(" + midX + "," + midY + ")";
                    })
                    .style("text-anchor", "middle")
                    .text(function(d) {
                        if (d.source.data.split_attribute === "outlook" && d.target.data.order === "sunny") {
                            return "sunny";
                        } else if (d.source.data.split_attribute === "outlook" && d.target.data.order === "yes") {
                            return "overcast";
                        } else if (d.source.data.split_attribute === "outlook" && d.target.data.order === "windy") {
                            return "rainy";
                        } else {
                            return d.target.data.order;
                        }
                    });
}

// Lấy dữ liệu JSON từ server
fetch('http://localhost:8000/test')
    .then(response => response.json())
    .then(jsonData => {
        // Vẽ cây quyết định
        drawTree(jsonData);
    })
    .catch(error => {
        console.error('Error:', error);
    });
