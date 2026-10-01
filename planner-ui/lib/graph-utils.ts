import { MarkerType, type Node, type Edge } from "@xyflow/react";
import type { PlannedWorkflow } from "./types";

const LAYER_GAP_Y = 150;
const NODE_GAP_X = 240;

/**
 * Convert a PlannedWorkflow into reactflow nodes and edges.
 *
 * Forward edges (depends_on that follow topological order) render as normal
 * blue smoothstep arrows top-to-bottom. Back edges (loop-back into an earlier
 * layer, e.g. a design refiner feeding into the sweep stager for the next
 * iteration) render as orange dashed animated arrows routed around the side.
 */
export function workflowToGraph(workflow: PlannedWorkflow): {
  nodes: Node[];
  edges: Edge[];
} {
  const agents = workflow.agents;
  const nameSet = new Set(agents.map((a) => a.agent_name));

  // Adjacency: parent -> children (based on depends_on, filtered to known agents)
  const dependents = new Map<string, string[]>();
  agents.forEach((a) => {
    a.depends_on
      .filter((dep) => nameSet.has(dep))
      .forEach((dep) => {
        const list = dependents.get(dep) || [];
        list.push(a.agent_name);
        dependents.set(dep, list);
      });
  });

  // Topological layer assignment
  const inDegree = new Map<string, number>();
  agents.forEach((a) => {
    const deps = a.depends_on.filter((d) => nameSet.has(d));
    inDegree.set(a.agent_name, deps.length);
  });

  const layers: string[][] = [];
  const assigned = new Set<string>();

  while (assigned.size < agents.length) {
    const layer: string[] = [];
    agents.forEach((a) => {
      if (!assigned.has(a.agent_name) && (inDegree.get(a.agent_name) || 0) === 0) {
        layer.push(a.agent_name);
      }
    });

    if (layer.length === 0) {
      // Cycle: pick the unassigned agent with the fewest remaining deps
      let best: string | null = null;
      let bestDeg = Infinity;
      agents.forEach((a) => {
        if (!assigned.has(a.agent_name)) {
          const d = inDegree.get(a.agent_name) || 0;
          if (d < bestDeg) {
            bestDeg = d;
            best = a.agent_name;
          }
        }
      });
      if (best) layer.push(best);
    }

    layers.push(layer);
    layer.forEach((name) => {
      assigned.add(name);
      (dependents.get(name) || []).forEach((child) => {
        inDegree.set(child, (inDegree.get(child) || 1) - 1);
      });
    });
  }

  // Build layer index lookup for back-edge detection
  const layerOf = new Map<string, number>();
  layers.forEach((layer, idx) => {
    layer.forEach((name) => layerOf.set(name, idx));
  });

  // Position nodes
  const nodes: Node[] = [];
  const maxLayerSize = Math.max(...layers.map((l) => l.length), 1);
  const graphWidth = maxLayerSize * NODE_GAP_X;

  layers.forEach((layer, layerIdx) => {
    const layerWidth = layer.length * NODE_GAP_X;
    const offsetX = (graphWidth - layerWidth) / 2;

    layer.forEach((name, nodeIdx) => {
      const agent = agents.find((a) => a.agent_name === name)!;
      nodes.push({
        id: name,
        type: "agentNode",
        position: {
          x: offsetX + nodeIdx * NODE_GAP_X,
          y: layerIdx * LAYER_GAP_Y,
        },
        data: {
          label: name,
          domain: agent.domain,
          mcpServers: agent.mcp_servers,
        },
      });
    });
  });

  // Build edges: classify each as forward or back
  const edges: Edge[] = [];
  agents.forEach((agent) => {
    const targetLayer = layerOf.get(agent.agent_name) ?? 0;
    agent.depends_on
      .filter((dep) => nameSet.has(dep))
      .forEach((dep) => {
        const sourceLayer = layerOf.get(dep) ?? 0;
        const isBackEdge = sourceLayer >= targetLayer;

        if (isBackEdge) {
          // Loop-back edge: route via right-side handles, orange + dashed
          edges.push({
            id: `${dep}->${agent.agent_name}`,
            source: dep,
            target: agent.agent_name,
            sourceHandle: "loop-out",
            targetHandle: "loop-in",
            type: "smoothstep",
            animated: true,
            label: "loop",
            labelStyle: { fill: "#c2410c", fontSize: 11, fontWeight: 600 },
            labelBgStyle: { fill: "#fff7ed" },
            style: {
              stroke: "#f97316",
              strokeWidth: 2,
              strokeDasharray: "6 4",
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: "#f97316",
              width: 20,
              height: 20,
            },
          });
        } else {
          // Forward edge: animated dashed blue arrow (top-to-bottom flow).
          // Skip-layer edges (source and target more than one layer apart)
          // use bezier curves so they arc gracefully around intermediate
          // nodes instead of doing ugly right-angle detours via smoothstep.
          const isSkipLayer = targetLayer - sourceLayer > 1;
          edges.push({
            id: `${dep}->${agent.agent_name}`,
            source: dep,
            target: agent.agent_name,
            type: isSkipLayer ? "bezier" : "smoothstep",
            animated: true,
            style: {
              stroke: "#4f6ef7",
              strokeWidth: 2,
              strokeDasharray: "6 4",
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: "#4f6ef7",
              width: 20,
              height: 20,
            },
          });
        }
      });
  });

  return { nodes, edges };
}
