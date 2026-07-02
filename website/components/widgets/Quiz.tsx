'use client';

import { useId, useState } from 'react';
import { ChevronDown } from 'lucide-react';

export interface QuizQuestion {
  question: string;
  options: string[];
  answer: number;
  explanation: string;
}

export function Quiz({ questions }: { questions: QuizQuestion[] }) {
  return (
    <details aria-label="章末測驗" className="group my-8 rounded-lg border border-border bg-card/40 p-5">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded text-base font-semibold text-foreground marker:content-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary [&::-webkit-details-marker]:hidden">
        <span aria-hidden>📝</span>
        <span>章末測驗</span>
        <ChevronDown
          aria-hidden
          className="ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180"
        />
      </summary>
      <ol className="mt-4 space-y-6">
        {questions.map((q, i) => (
          <QuizItem key={i} index={i} question={q} />
        ))}
      </ol>
    </details>
  );
}

function QuizItem({ index, question }: { index: number; question: QuizQuestion }) {
  const [selected, setSelected] = useState<number | null>(null);
  const groupId = useId();
  const answered = selected !== null;
  const isCorrect = selected === question.answer;

  return (
    <li className="list-none">
      <fieldset>
        <legend className="mb-2 font-medium text-foreground">
          {index + 1}. {question.question}
        </legend>
        <div className="space-y-2" role="radiogroup" aria-label={`第 ${index + 1} 題選項`}>
          {question.options.map((option, oi) => {
            const state =
              answered && oi === question.answer
                ? 'border-primary bg-primary/10 text-foreground'
                : answered && oi === selected
                  ? 'border-[#f85149] bg-[#f85149]/10 text-foreground'
                  : 'border-border hover:border-border';
            return (
              <label
                key={oi}
                className={`flex cursor-pointer items-center gap-3 rounded-md border px-3 py-2 text-sm transition ${state}`}
              >
                <input
                  type="radio"
                  name={groupId}
                  className="accent-brand"
                  checked={selected === oi}
                  onChange={() => setSelected(oi)}
                />
                <span>{option}</span>
              </label>
            );
          })}
        </div>
      </fieldset>
      {answered && (
        <p
          className={`mt-2 rounded-md px-3 py-2 text-sm ${
            isCorrect ? 'bg-primary/10 text-primary' : 'bg-[#f85149]/10 text-[#ff7b72]'
          }`}
          role="status"
        >
          {isCorrect ? '答對了!' : '再想想。'} {question.explanation}
        </p>
      )}
    </li>
  );
}
