from llama_index.core.schema import IndexNode, TransformComponent


class TextLinkToSource(TransformComponent):
    def __call__(self, nodes, **kwargs):
        return [
            IndexNode.from_text_node(node, node.source_node.node_id) for node in nodes
        ]
