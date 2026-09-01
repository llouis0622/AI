<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps<{
  id: string
  q: string
}>()

const revealed = ref(false)
const status = ref<'' | 'pass' | 'again'>('')

const storageKey = `aihub-quiz:${props.id}`

function loadStatus() {
  try {
    const v = localStorage.getItem(storageKey)
    if (v === 'pass' || v === 'again') status.value = v
  } catch {
    /* storage unavailable */
  }
}

function mark(v: 'pass' | 'again') {
  status.value = v
  try {
    localStorage.setItem(storageKey, v)
    window.dispatchEvent(new CustomEvent('aihub-quiz-change'))
  } catch {
    /* storage unavailable */
  }
}

onMounted(loadStatus)
</script>

<template>
  <div class="quiz" :class="{ pass: status === 'pass', again: status === 'again' }">
    <div class="quiz-q">
      <span class="quiz-badge" aria-hidden="true">Q</span>
      <span class="quiz-q-text">{{ q }}</span>
      <span v-if="status === 'pass'" class="quiz-status">맞힘 ✓</span>
      <span v-else-if="status === 'again'" class="quiz-status">복습 필요 ↻</span>
    </div>
    <button v-if="!revealed" class="quiz-reveal" @click="revealed = true">정답 보기</button>
    <div v-else>
      <div class="quiz-a">
        <slot />
      </div>
      <div class="quiz-actions">
        <button class="quiz-btn quiz-btn-pass" @click="mark('pass')">맞혔다 ✓</button>
        <button class="quiz-btn quiz-btn-again" @click="mark('again')">다시 복습 ↻</button>
        <button class="quiz-btn" @click="revealed = false">답 숨기기</button>
      </div>
    </div>
  </div>
</template>
