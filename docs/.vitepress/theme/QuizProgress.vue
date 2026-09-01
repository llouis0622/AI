<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  prefix: string
  total: number
}>()

const passed = ref(0)
const again = ref(0)

function recount() {
  let p = 0
  let a = 0
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key && key.startsWith(`aihub-quiz:${props.prefix}-`)) {
        const v = localStorage.getItem(key)
        if (v === 'pass') p++
        else if (v === 'again') a++
      }
    }
  } catch {
    /* storage unavailable */
  }
  passed.value = p
  again.value = a
}

function reset() {
  try {
    const keys: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key && key.startsWith(`aihub-quiz:${props.prefix}-`)) keys.push(key)
    }
    keys.forEach((k) => localStorage.removeItem(k))
    window.dispatchEvent(new CustomEvent('aihub-quiz-change'))
  } catch {
    /* storage unavailable */
  }
  recount()
}

onMounted(() => {
  recount()
  window.addEventListener('aihub-quiz-change', recount)
})
onUnmounted(() => {
  window.removeEventListener('aihub-quiz-change', recount)
})
</script>

<template>
  <div class="quiz-progress">
    <div class="quiz-progress-bar">
      <div
        class="quiz-progress-fill"
        :style="{ width: `${total > 0 ? Math.round((passed / total) * 100) : 0}%` }"
      />
    </div>
    <div class="quiz-progress-text">
      <strong>{{ passed }}</strong> / {{ total }} 문제 맞힘
      <span v-if="again > 0"> · 복습 필요 {{ again }}문제</span>
      <button class="quiz-btn quiz-progress-reset" @click="reset">기록 초기화</button>
    </div>
  </div>
</template>
