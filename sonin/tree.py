from typing import Any, Callable, Generator

from pydantic import BaseModel

from sonin.sonin_random import rand_bool


class BinaryTree(BaseModel):
    root: Any | None
    is_leaf: Callable[[Any], bool]

    def __iter__(self) -> Generator[Any, None, None]:
        if self.root is not None:
            if self.is_leaf(self.root):
                yield self.root
            else:
                yield from self.root

    def branches_iter(self) -> Generator[Any, None, None]:
        if self.root is not None and not self.is_leaf(self.root):
            yield from self.root.branches_iter()

    def find_child_and_parents(self, is_next_left: Callable[[], bool] = rand_bool):
        child = self.root
        parent = None
        g_parent = None
        was_left = None
        g_was_left = None

        if child is None:
            return None, None, None, None, None
        elif self.is_leaf(child):
            return child, None, None, None, None

        while not self.is_leaf(child):
            g_parent = parent
            parent = child
            g_was_left = was_left
            was_left = is_next_left()

            if was_left:
                child = parent.left
            else:
                child = parent.right

        return child, parent, g_parent, was_left, g_was_left

    def add(self, leaf: Any, new_branch: Callable[[Any, Any], Any], is_next_left: Callable[[], bool] = rand_bool):
        if self.root is None:
            self.root = leaf
            return

        child, parent, _, was_left, _ = self.find_child_and_parents(is_next_left)

        def left_right(l, r):
            if is_next_left():
                return l, r
            else:
                return r, l

        if child is self.root:
            left, right = left_right(leaf, self.root)
            self.root = new_branch(left, right)
        elif was_left:
            left, right = left_right(leaf, parent.left)
            parent.left = new_branch(left, right)
        else:
            left, right = left_right(leaf, parent.right)
            parent.right = new_branch(left, right)

    def remove(self, is_next_left: Callable[[], bool] = rand_bool):
        if self.root is None:
            return
        elif self.is_leaf(self.root):
            self.root = None
            return

        _, parent, g_parent, was_left, g_was_left = self.find_child_and_parents(is_next_left)

        if parent is self.root:
            if was_left:
                self.root = parent.right
            else:
                self.root = parent.left
        elif g_was_left:
            if was_left:
                g_parent.left = parent.right
            else:
                g_parent.left = parent.left
        else:
            if was_left:
                g_parent.right = parent.right
            else:
                g_parent.right = parent.left
