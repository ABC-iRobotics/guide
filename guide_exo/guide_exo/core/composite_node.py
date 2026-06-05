from typing import Dict
from typing import List
from typing import Optional

from guide_exo.core.states import Layer
from guide_exo.core.states import DemoStatus
from guide_exo.core.states import ExecutionResult
from guide_exo.core.states import ExecutionMode

from guide_exo.core.base_node import BaseNode

class CompositeNode(BaseNode):
    def __init__(self, 
                 name: str, 
                 level: Layer, 
                 children: Optional[List[BaseNode]] = None, 
                 fallbacks: Dict[str, 'RecoveryNode'] = None, 
                 overrides: Dict[str, BaseNode] = None,
                 
                 mode: str = "normal",                       
                 condition_expr: Optional[str] = None,       
                 true_branch: Optional[BaseNode] = None,     
                 false_branch: Optional[BaseNode] = None,    
                 max_loops: int = 10,                        
                 **kwargs):
        
        super().__init__(name, **kwargs)
        self.level = level
        self.children: List[BaseNode] = children or []
        self.fallbacks: Dict[str, 'RecoveryNode'] = fallbacks or {}
        
        self.mode = ExecutionMode(mode)
        self.condition_expr = condition_expr
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.max_loops = max_loops

        # VALIDATION: Children must be strictly lower
        for child in self.children:
            if child.level != Layer.UTILITY and child.level >= self.level:
                raise ValueError(
                    f"[Architecture Violation] '{self.name}' ({self.level.name}) "
                    f"cannot contain child '{child.name}' ({child.level.name})."
                )
                
        # VALIDATION: Recovery Node can be up to the SAME level as the parent
        for recovery_node in self.fallbacks.values():
            if recovery_node.level > self.level:
                raise ValueError(
                    f"[Architecture Violation] Recovery '{recovery_node.name}' ({recovery_node.level.name}) "
                    f"cannot be higher than its container '{self.name}' ({self.level.name})."
                )

        # VALIDATION: Branches can be up to the SAME level as the parent
        for branch, branch_name in [(self.true_branch, "true_branch"), (self.false_branch, "false_branch")]:
            if branch and branch.level > self.level:
                raise ValueError(
                    f"[Architecture Violation] Branch '{branch_name}' ({branch.level.name}) "
                    f"cannot be higher than its container '{self.name}' ({self.level.name})."
                )
        
        # VALIDATION: Override Nodes must be strictly lower or same level
        if overrides is not None:
            for node in overrides.values():
                if node.level > self.level:
                    raise ValueError(
                        f"[Architecture Violation] Override '{node.name}' ({node.level.name}) "
                        f"cannot be higher than its container '{self.name}' ({self.level.name})."
                    )

        self._apply_override(overrides)

    def _get_child_index(self, name: str) -> int:
        for i, child in enumerate(self.children):
            if child.name == name: return i
        return -1

    def _extract_scoped(self, target_name: str, overrides: Dict[str, BaseNode]) -> Dict[str, BaseNode]:
        """Kivonja és megtisztítja az adott node-ra vonatkozó pont-notációs override-okat."""
        prefix = f"{target_name}."
        return {k.removeprefix(prefix): v for k, v in overrides.items() if k.startswith(prefix)}

    def _apply_override(self, overrides: Dict[str, BaseNode]):
        """Rekurzívan alkalmazza a felülírásokat a belső struktúrákra."""
        if not overrides:
            return

        # 1. Children felülírása
        for i, child in enumerate(self.children):
            if overrides.get(child.name):
                self.children[i] = overrides[child.name]
            self.children[i]._apply_override(self._extract_scoped(self.children[i].name, overrides))

        # 2. Fallbacks felülírása
        for key, fallback in list(self.fallbacks.items()):
            if overrides.get(fallback.name):
                self.fallbacks[key] = overrides[fallback.name]
            self.fallbacks[key]._apply_override(self._extract_scoped(self.fallbacks[key].name, overrides))

        # 3. Branches felülírása
        for branch_attr in ['true_branch', 'false_branch']:
            branch = getattr(self, branch_attr)
            if branch:
                if overrides.get(branch.name):
                    setattr(self, branch_attr, overrides[branch.name])
                # Frissített referencia lekérése a rekurzióhoz
                getattr(self, branch_attr)._apply_override(self._extract_scoped(getattr(self, branch_attr).name, overrides))

    def _evaluate_condition(self, context: dict, iteration: int) -> bool:
        """Safely evaluates the condition expression using the provided context."""
        if not self.condition_expr:
            return False
        eval_context = {**context, '_iteration': iteration}
        try:
            return bool(eval(self.condition_expr, {"__builtins__": {}}, eval_context))
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to evaluate condition '{self.condition_expr}': {e}")
            raise

    def run(self, **kwargs) -> ExecutionResult:
        self.logger.info(f"\n[STARTING] {self.level.name}: {self.name} (Mode: {self.mode.value})")
        
        local_context = {**kwargs}
        retry_counts = {child.name: 0 for child in self.children}
        
        current_iteration = 0
        loop_limit = self.max_loops if self.mode == ExecutionMode.LOOP else 1

        while current_iteration < loop_limit:
            
            # --- 1. CHILDREN EXECUTION (Safely skips if empty) ---
            if self.children:
                self.logger.debug(f"[{self.name}] Executing children sequence (Iteration {current_iteration})")
            
            current_index = 0
            sequence_failed = False
            
            while current_index < len(self.children):
                child = self.children[current_index]
                
                result = child.execute(local_context)
                local_context.update(result.outputs)
                
                # Error Handling & Fallback logic
                if result.status == DemoStatus.FAILURE:
                    if child.name in self.fallbacks:
                        recovery = self.fallbacks[child.name]
                        if retry_counts[child.name] >= recovery.max_retries:
                            self.logger.error(f"  [FATAL] '{child.name}' failed. Max retries exceeded.")
                            return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)
                            
                        retry_counts[child.name] += 1
                        self.logger.warning(f"  [WARN] '{child.name}' failed. Executing Recovery: {recovery.name}...")
                        
                        recovery_res = recovery.execute(local_context)
                        if recovery_res.status == DemoStatus.FAILURE:
                            return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)
                        
                        local_context.update(recovery_res.outputs)
                        
                        resume_idx = self._get_child_index(recovery.resume_target)
                        if resume_idx != -1:
                            current_index = resume_idx
                            continue
                        else:
                            self.logger.error(f"  [FATAL] Recovery target '{recovery.resume_target}' not found.")
                            return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)
                    else:
                        self.logger.error(f"  [FATAL] '{child.name}' failed with no fallback. Sequence aborted.")
                        sequence_failed = True
                        break
                
                current_index += 1

            # Abort composite execution if a child failed without recovery
            if sequence_failed:
                return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)

            # --- 2. BLOCK-LEVEL EVALUATION (Mode Specific) ---
            
            if self.mode == ExecutionMode.NORMAL:
                return ExecutionResult(DemoStatus.PERFECT, outputs=local_context)

            elif self.mode == ExecutionMode.CONDITION:
                if not self.condition_expr:
                    self.logger.error(f"[{self.name}] Mode is CONDITION but no condition_expr provided.")
                    return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)

                try:
                    is_true = self._evaluate_condition(local_context, current_iteration)
                    self.logger.info(f"[{self.name}] Condition '{self.condition_expr}' evaluated to {is_true}")
                except Exception:
                    return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)

                # Execute the corresponding branch if provided
                branch_node = self.true_branch if is_true else self.false_branch
                if branch_node:
                    self.logger.info(f"[{self.name}] Executing branch node: {branch_node.name}")
                    branch_res = branch_node.execute(local_context)
                    local_context.update(branch_res.outputs)
                    
                    if branch_res.status != DemoStatus.PERFECT:
                         return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)

                return ExecutionResult(DemoStatus.PERFECT, outputs=local_context)

            elif self.mode == ExecutionMode.LOOP:
                if self.condition_expr:
                    try:
                        exit_condition = self._evaluate_condition(local_context, current_iteration)
                    except Exception:
                        return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)

                    if exit_condition:
                        self.logger.info(f"[{self.name}] Loop exit condition met.")
                        return ExecutionResult(DemoStatus.PERFECT, outputs=local_context)
                
                if current_iteration + 1 >= self.max_loops:
                    self.logger.error(f"[{self.name}] Max loops ({self.max_loops}) reached. Aborting loop.")
                    return ExecutionResult(DemoStatus.FAILURE, outputs=local_context)

                self.logger.debug(f"[{self.name}] Condition false. Continuing loop...")

            current_iteration += 1

        return ExecutionResult(DemoStatus.PERFECT, outputs=local_context)
  
  
    
class RecoveryNode(CompositeNode):
    def __init__(self, 
                 name: str, 
                 level: Layer, 
                 children: List[BaseNode], 
                 resume_target: str, 
                 max_retries: int = 3, 
                 fallbacks: Dict[str, 'RecoveryNode'] = None, 
                 **kwargs):
        super().__init__(name, level, children, fallbacks, **kwargs)
        self.resume_target = resume_target
        self.max_retries = max_retries