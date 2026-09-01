import DefaultTheme from 'vitepress/theme'
import Quiz from './Quiz.vue'
import QuizProgress from './QuizProgress.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Quiz', Quiz)
    app.component('QuizProgress', QuizProgress)
  }
}
