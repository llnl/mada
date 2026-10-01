"use client";

import { useCallback, useEffect, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type { PlannedWorkflow } from "@/lib/types";
import { workflowToGraph } from "@/lib/graph-utils";
import AgentNode from "./AgentNode";

interface WorkflowGraphProps {
  workflow: PlannedWorkflow;
  selectedAgent: string | null;
  onSelectAgent: (agentName: string | null) => void;
}

const nodeTypes: NodeTypes = {
  agentNode: AgentNode,
};

/**
 * Inner graph component. Wrapped by ReactFlowProvider below so we can
 * call useReactFlow() to trigger fitView after the workflow changes.
 */
function WorkflowGraphInner({
  workflow,
  selectedAgent,
  onSelectAgent,
}: WorkflowGraphProps) {
  const { nodes: computedNodes, edges: computedEdges } = useMemo(
    () => workflowToGraph(workflow),
    [workflow]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(computedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(computedEdges);
  const { fitView } = useReactFlow();

  // React Flow's state hooks only take an initial value once; when the
  // workflow changes (regenerate, refine, agent added/removed), we have to
  // manually push the new nodes/edges into their state.
  useEffect(() => {
    setNodes(computedNodes);
    setEdges(computedEdges);
    // Re-fit the viewport so newly added nodes are visible.
    // requestAnimationFrame gives React Flow time to lay out the new nodes.
    requestAnimationFrame(() => {
      fitView({ padding: 0.3, duration: 300 });
    });
  }, [computedNodes, computedEdges, setNodes, setEdges, fitView]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      onSelectAgent(node.id === selectedAgent ? null : node.id);
    },
    [selectedAgent, onSelectAgent]
  );

  const onPaneClick = useCallback(() => {
    onSelectAgent(null);
  }, [onSelectAgent]);

  return (
    <ReactFlow
      nodes={nodes.map((n) => ({
        ...n,
        selected: n.id === selectedAgent,
      }))}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      onPaneClick={onPaneClick}
      fitView
      fitViewOptions={{ padding: 0.3 }}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#e5e7eb" gap={16} />
      <Controls />
    </ReactFlow>
  );
}

export default function WorkflowGraph(props: WorkflowGraphProps) {
  return (
    <div className="h-[70vh] min-h-[500px] w-full rounded-lg border border-gray-200 bg-white">
      <ReactFlowProvider>
        <WorkflowGraphInner {...props} />
      </ReactFlowProvider>
    </div>
  );
}
