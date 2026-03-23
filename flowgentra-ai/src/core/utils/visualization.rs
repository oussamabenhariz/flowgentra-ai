//! # Graph Visualization Module
//!
//! Generates clear, professional SVG visualizations of agent workflows with crisp, scalable text.

use crate::core::error::Result;

use std::collections::HashMap;
use svg;

/// Configuration for graph visualization  
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Output file path for SVG
    pub output_path: String,
}

impl VisualizationConfig {
    /// Create new visualization config
    pub fn new(output_path: impl Into<String>) -> Self {
        Self {
            output_path: output_path.into(),
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self::new("agent_graph.svg")
    }
}

/// Node layout information
#[derive(Clone)]
struct Node {
    id: String,
    x: f32,
    y: f32,
    radius: f32,
}

/// Calculate node position and determine layout using topological sort
fn compute_layout<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>) -> (Vec<Node>, f32, f32) {
    let nodes = graph.nodes();
    let edges = graph.edges();

    if nodes.is_empty() {
        return (Vec::new(), 800.0, 600.0);
    }

    // Build adjacency info for topological sort
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut out_edges: HashMap<String, Vec<String>> = HashMap::new();

    for (node_id, _) in nodes.iter() {
        in_degree.insert(node_id.clone(), 0); // Add explicit type annotation if needed
        out_edges.insert(node_id.clone(), Vec::new());
    }

    for edge in edges {
        *in_degree.entry(edge.to.clone()).or_insert(0) += 1;
        out_edges
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
    }

    // Assign initial levels (0). We'll then relax edges to push downstream nodes
    // to larger levels. This handles cycles gracefully by iterating enough times.
    let mut levels: HashMap<String, usize> = HashMap::new();
    for (node_id, _) in nodes.iter() {
        levels.insert(node_id.clone(), 0usize); // Add explicit type annotation if needed
    }

    // Seed from nodes with no incoming edges for better hierarchy
    for (node_id, &deg) in &in_degree {
        if deg == 0 {
            levels.insert(node_id.clone(), 0);
        }
    }

    // Relaxation: for up to N iterations, propagate level = max(pred_level + 1)
    let n = nodes.len().max(1);
    for _ in 0..(n + 2) {
        for edge in edges.iter() {
            let from_level = *levels.get(&edge.from).unwrap_or(&0);
            let to_level = *levels.get(&edge.to).unwrap_or(&0);
            if from_level + 1 > to_level {
                levels.insert(edge.to.clone(), from_level + 1);
            }
        }
    }

    // Group nodes by level
    let max_level = levels.values().max().copied().unwrap_or(0);
    let mut level_nodes: Vec<Vec<(String, f32)>> = vec![Vec::new(); max_level + 1];

    for (node_id, _) in nodes.iter() {
        let level = levels[node_id];
        let radius = calculate_node_radius(node_id);
        level_nodes[level].push((node_id.to_string(), radius));
    }

    // Calculate positions
    let mut layout = Vec::new();
    let padding = 60.0;
    let node_spacing = 100.0;
    let level_spacing = 150.0;

    // We'll compute per-level widths and track the rightmost extent to pick
    // a suitable canvas width so nodes are centered and don't overflow.
    let mut max_right = 0.0f32;
    let mut max_y = 0.0f32;

    for (level, nodes_at_level) in level_nodes.iter().enumerate() {
        let y = padding + level as f32 * level_spacing;

        // Calculate total width needed for this level
        let total_width: f32 = nodes_at_level
            .iter()
            .map(|(_, r)| r * 2.0 + node_spacing)
            .sum();
        let mut start_x = padding;

        // If total width is smaller than a central target, center it
        let center_target = 800.0;
        if total_width + padding * 2.0 < center_target {
            start_x = center_target / 2.0 - (total_width / 2.0);
        }

        let mut x = start_x;
        for (node_id, radius) in nodes_at_level {
            let node_x = x + radius;
            layout.push(Node {
                id: node_id.clone(),
                x: node_x,
                y,
                radius: *radius,
            });
            x += radius * 2.0 + node_spacing;
        }

        // update extents
        if !nodes_at_level.is_empty() {
            let right = x - node_spacing + padding;
            if right > max_right {
                max_right = right;
            }
            if y > max_y {
                max_y = y;
            }
        }
    }

    // Ensure minimal extents
    let canvas_w = (max_right + padding).max(800.0);
    let canvas_h = (max_y + 200.0).max(600.0);

    (layout, canvas_w, canvas_h)
}

/// Calculate node radius based on label length
fn calculate_node_radius(label: &str) -> f32 {
    let char_count = label.len() as f32;
    // Minimum 30, increases with label length
    (30.0 + char_count * 3.0).min(70.0)
}

/// Generate SVG visualization of agent graph with crisp, scalable text
pub fn visualize_graph<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>, config: VisualizationConfig) -> Result<()> {
    visualize_graph_inner(graph, &config, None)
}

/// Options for execution overlay
#[derive(Debug, Clone, Default)]
pub struct ExecutionOverlay {
    /// Node names in execution order (e.g. from ExecutionTrace::execution_path())
    pub path: Vec<String>,
}

/// Generate SVG with execution path highlighted
pub fn visualize_graph_with_execution<T: crate::core::state::State>(
    graph: &crate::core::graph::Graph<T>,
    config: VisualizationConfig,
    overlay: ExecutionOverlay,
) -> Result<()> {
    visualize_graph_inner(graph, &config, Some(overlay.path))
}

fn visualize_graph_inner<T: crate::core::state::State>(
    graph: &crate::core::graph::Graph<T>,
    config: &VisualizationConfig,
    exec_path: Option<Vec<String>>,
) -> Result<()> {
    let path_set: std::collections::HashSet<String> = exec_path
        .as_ref()
        .map(|p| p.iter().cloned().collect())
        .unwrap_or_default();
    let path_vec = exec_path.as_deref().unwrap_or(&[]);

    let (layout, min_w, min_h) = compute_layout(graph);
    let canvas_width = (min_w as i32).max(800);
    let canvas_height = (min_h as i32).max(600);

    let position_map: HashMap<String, (f32, f32, f32)> = layout
        .iter()
        .map(|n| (n.id.clone(), (n.x, n.y, n.radius)))
        .collect();

    let mut document = svg::Document::new()
        .set("viewBox", format!("0 0 {} {}", canvas_width, canvas_height))
        .set("width", canvas_width)
        .set("height", canvas_height)
        .set("style", "background-color: white; font-family: sans-serif;");

    let style = svg::node::element::Style::new(
        ".node-circle { fill: #64b4dc; stroke: #3278b4; stroke-width: 2.5; }\
         .node-executed { fill: #22c55e; stroke: #15803d; stroke-width: 3; }\
         .shadow-circle { fill: #c8c8c8; opacity: 0.4; }\
         .edge-line { stroke: #64788c; stroke-width: 2; }\
         .edge-executed { stroke: #22c55e; stroke-width: 3; }\
         .arrow-head { fill: #64788c; }\
         .arrow-executed { fill: #22c55e; }\
         .node-text { fill: white; font-size: 16px; text-anchor: middle; dominant-baseline: central; font-weight: 600; }\
         .edge-label { fill: #505050; font-size: 13px; text-anchor: middle; }"
    );
    document = document.add(style);

    // Draw edges first (behind nodes)
    for edge in graph.edges() {
        if let (Some(&(x1, y1, r1)), Some(&(x2, y2, r2))) =
            (position_map.get(&edge.from), position_map.get(&edge.to))
        {
            let dx = x2 - x1;
            let dy = y2 - y1;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist > 0.1 {
                let ux = dx / dist;
                let uy = dy / dist;

                let start_x = x1 + ux * r1;
                let start_y = y1 + uy * r1;
                let end_x = x2 - ux * r2;
                let end_y = y2 - uy * r2;

                let edge_executed = path_vec
                    .windows(2)
                    .any(|w| w[0] == edge.from && w[1] == edge.to);
                let edge_class = if edge_executed {
                    "edge-executed"
                } else {
                    "edge-line"
                };

                let line = svg::node::element::Line::new()
                    .set("x1", format!("{:.1}", start_x))
                    .set("y1", format!("{:.1}", start_y))
                    .set("x2", format!("{:.1}", end_x))
                    .set("y2", format!("{:.1}", end_y))
                    .set("class", edge_class);
                document = document.add(line);

                let arrow_class = if edge_executed {
                    "arrow-executed"
                } else {
                    "arrow-head"
                };

                // Draw arrowhead
                let arrow_size = 15.0;
                let arrow_x = end_x - ux * arrow_size;
                let arrow_y = end_y - uy * arrow_size;
                let perp_x = -uy;
                let perp_y = ux;

                let p1_x = end_x;
                let p1_y = end_y;
                let p2_x = arrow_x + perp_x * 8.0;
                let p2_y = arrow_y + perp_y * 8.0;
                let p3_x = arrow_x - perp_x * 8.0;
                let p3_y = arrow_y - perp_y * 8.0;

                let points = format!(
                    "{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}",
                    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
                );
                let poly = svg::node::element::Polygon::new()
                    .set("points", points)
                    .set("class", arrow_class);
                document = document.add(poly);

                // Draw condition label if present
                if let Some(ref cond) = edge.condition_name {
                    let label_x = (start_x + end_x) / 2.0;
                    let label_y = (start_y + end_y) / 2.0 - 15.0;
                    let text = svg::node::element::Text::new()
                        .set("x", format!("{:.1}", label_x))
                        .set("y", format!("{:.1}", label_y))
                        .set("class", "edge-label")
                        .add(svg::node::Text::new(cond.clone()));
                    document = document.add(text);
                }
            }
        }
    }

    // Draw nodes with shadows and crisp text
    for node in &layout {
        // Draw shadow circle for depth
        let shadow = svg::node::element::Circle::new()
            .set("cx", format!("{:.1}", node.x - 2.0))
            .set("cy", format!("{:.1}", node.y + 3.0))
            .set("r", format!("{:.1}", node.radius + 2.0))
            .set("class", "shadow-circle");
        document = document.add(shadow);

        let node_executed = path_set.contains(&node.id);
        let node_class = if node_executed {
            "node-executed"
        } else {
            "node-circle"
        };

        let circle = svg::node::element::Circle::new()
            .set("cx", format!("{:.1}", node.x))
            .set("cy", format!("{:.1}", node.y))
            .set("r", format!("{:.1}", node.radius))
            .set("class", node_class);
        document = document.add(circle);

        // Draw node label text using SVG <text> (always crisp and scalable)
        let text = svg::node::element::Text::new()
            .set("x", format!("{:.1}", node.x))
            .set("y", format!("{:.1}", node.y))
            .set("class", "node-text")
            .add(svg::node::Text::new(node.id.clone()));
        document = document.add(text);
    }

    // Save SVG to file
    svg::save(&config.output_path, &document)
        .map_err(|_| crate::core::error::FlowgentraError::GraphError("Failed to save SVG".into()))?;

    tracing::info!(output_path = %config.output_path, "Graph visualization generated ({}x{})", canvas_width, canvas_height);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visualization_config_defaults() {
        let config = VisualizationConfig::default();
        assert_eq!(config.output_path, "agent_graph.svg");
    }

    #[test]
    fn visualization_config_custom() {
        let config = VisualizationConfig::new("my_graph.svg");
        assert_eq!(config.output_path, "my_graph.svg");
    }

    #[test]
    fn test_node_radius_calculation() {
        assert_eq!(calculate_node_radius("a"), 33.0);
        assert_eq!(calculate_node_radius("longer_name"), 63.0);
        assert_eq!(calculate_node_radius("very_very_long_name_here"), 70.0); // Capped at 70
    }
}
