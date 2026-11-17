; ============================================================
;  SKYNET.ASM - AI Agent TSR for 8088
;  Assembler: TASM / MASM / tinyasm (nanochess)
;  Format: COM file (Tiny model, < 64KB)
;  
;  Features:
;  - Loads neural network weights from SKYNET.WGT
;  - Hooks timer interrupt for periodic "thinking"
;  - Monitors system state via sensors
;  - Takes actions based on NN output
;  - Hotkey activation: ALT+S (show status)
; ============================================================

CODE SEGMENT
    ORG 100h            ; COM файл начинается с 100h

START:
    jmp install         ; Переходим к установке TSR

; ============================================================
;  DATA SECTION (после JMP, чтобы не выполнялось)
; ============================================================

; --- Строки сообщений ---
msg_banner   db 13,10,'SKYNET AI Agent v0.1',13,10
             db '======================',13,10,'$'
msg_loading db 'Loading neural weights...',13,10,'$'
msg_ok      db 'SKYNET installed.',13,10
             db 'Memory used: $'
msg_bytes   db ' bytes',13,10,'$'
msg_hotkey  db 13,10,'Press ALT-S for status',13,10,'$'
msg_errfile db 'ERROR: Cannot open SKYNET.WGT!',13,10,'$'
msg_errmem  db 'ERROR: Not enough memory!',13,10,'$'

; --- Имя файла весов ---
fname_wgt    db 'SKYNET.WGT',0

; --- Константы сети ---
INPUT_SIZE   equ 8
HIDDEN_SIZE  equ 12
OUTPUT_SIZE  equ 4
FP_SHIFT     equ 8
FP_ONE       equ 256

; --- Сигмоид LUT (256 байт, предрассчитан) ---
sigmoid_lut:
    db 2,2,2,3,3,3,3,4,4,4,5,5,5,6,6,7
    db 7,8,9,10,11,12,13,14,16,17,19,21,23,25,27,30
    db 33,36,39,43,47,51,56,61,67,73,79,86,94,102,111,120
    db 130,141,152,164,177,190,204,218,233,248,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
    db 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255

; --- Переменные агента ---
; (располагаем после кода, в конце COM-файла)

; Веса нейросети (загружаются из файла)
w_ih:  dw INPUT_SIZE*HIDDEN_SIZE dup (?)   ; ~192 bytes
w_ho:  dw HIDDEN_SIZE*OUTPUT_SIZE dup (?)  ; ~48 bytes
b_h:   dw HIDDEN_SIZE dup (?)               ; ~24 bytes
b_o:   dw OUTPUT_SIZE dup (?)               ; ~8 bytes

; Состояние сенсоров (обновляются по таймеру)
sensors: dw INPUT_SIZE dup (?)              ; ~16 bytes

; Активации (временные при инференсе)
hidden_act: dw HIDDEN_SIZE dup (?)          ; ~24 bytes
output_act: dw OUTPUT_SIZE dup (?)          ; ~8 bytes

; Состояние агента
agent_state: db 0                           ; 0=idle, 1=thinking, 2=acting
tick_counter: dw 0                           ; Счётчик тиков
think_interval: dw 18                        ; Думать раз в секунду (18.2 тика)
action_cooldown: dw 0                        ; Кулдаун между действиями

; Старые векторы прерываний (для восстановления)
old_int08: dd ?
old_int09: dd ?
old_int28: dd ?

; Признак установки (для предотвращения повторной установки)
installed_flag: db 0

; ============================================================
;  INSTALL ROUTINE - Установка TSR в память
; ============================================================
install:
    ; Проверка, не установлен ли уже
    mov al, [installed_flag]
    cmp al, 0FFh
    je already_installed
    
    ; Вывод баннера
    mov dx, offset msg_banner
    mov ah, 09h
    int 21h
    
    ; Загрузка весов
    mov dx, offset msg_loading
    mov ah, 09h
    int 21h
    
    call load_weights
    jc load_error
    
    ; Сохранение старых векторов прерываний
    mov ax, 3508h             ; Получить INT 08h
    int 21h
    mov word ptr old_int08, bx
    mov word ptr old_int08+2, es
    
    mov ax, 3509h             ; Получить INT 09h
    int 21h
    mov word ptr old_int09, bx
    mov word ptr old_int09+2, es
    
    mov ax, 3528h             ; Получить INT 28h
    int 21h
    mov word ptr old_int28, bx
    mov word ptr old_int28+2, es
    
    ; Установка новых обработчиков
    push ds
    mov dx, offset new_int08
    mov ax, 2508h
    int 21h
    
    mov dx, offset new_int09
    mov ax, 2509h
    int 21h
    
    mov dx, offset new_int28
    mov ax, 2528h
    int 21h
    pop ds
    
    ; Вычислить размер резидентной части
    mov dx, offset end_of_program
    inc dx                      ; Округление до параграфа
    mov cl, 4
    shr dx, cl                 ; Преобразовать в параграфы
    
    ; Вывод информации о памяти
    push dx
    mov dx, offset msg_ok
    mov ah, 09h
    int 21h
    
    pop ax                     ; AX = параграфы
    mov bx, ax
    shl bx, 4                  ; BX = байты
    call print_word_dec
    mov dx, offset msg_bytes
    mov ah, 09h
    int 21h
    
    mov dx, offset msg_hotkey
    mov ah, 09h
    int 21h
    
    ; Установить флаг
    mov byte ptr [installed_flag], 0FFh
    
    ; TSR - завершить, оставшись резидентным
    mov ah, 31h
    int 21h

already_installed:
    mov dx, offset msg_already
    mov ah, 09h
    int 21h
    mov ax, 4C00h
    int 21h

load_error:
    mov dx, offset msg_errfile
    mov ah, 09h
    int 21h
    mov ax, 4C01h
    int 21h

msg_already: db 'SKYNET already installed!',13,10,'$'

; ============================================================
;  LOAD_WEIGHTS - Загрузка весов из файла
;  Вход: ничего
;  Выход: CF=1 если ошибка
; ============================================================
load_weights proc near
    mov ax, 3D00h             ; Открыть для чтения
    mov dx, offset fname_wgt
    int 21h
    jc load_err               ; Ошибка открытия
    mov bx, ax                ; BX = handle
    
    ; Проверка magic number
    sub sp, 2
    mov cx, 2
    mov dx, sp
    mov ah, 3Fh
    int 21h
    pop ax
    cmp ax, 0x534E            ; 'SN'?
    jne load_err_close
    
    ; Пропустить архитектуру (3 байта)
    mov ah, 3Fh
    mov cx, 3
    sub sp, 3
    mov dx, sp
    int 21h
    add sp, 3
    
    ; Загрузка w_ih
    mov ah, 3Fh
    mov cx, INPUT_SIZE*HIDDEN_SIZE*2
    mov dx, offset w_ih
    int 21h
    jc load_err_close
    
    ; Загрузка w_ho
    mov ah, 3Fh
    mov cx, HIDDEN_SIZE*OUTPUT_SIZE*2
    mov dx, offset w_ho
    int 21h
    jc load_err_close
    
    ; Загрузка b_h
    mov ah, 3Fh
    mov cx, HIDDEN_SIZE*2
    mov dx, offset b_h
    int 21h
    jc load_err_close
    
    ; Загрузка b_o
    mov ah, 3Fh
    mov cx, OUTPUT_SIZE*2
    mov dx, offset b_o
    int 21h
    jc load_err_close
    
    ; Закрыть файл
    mov ah, 3Eh
    int 21h
    clc
    ret

load_err_close:
    mov ah, 3Eh
    int 21h
load_err:
    stc
    ret
load_weights endp

; ============================================================
;  INTERRUPT HANDLERS
; ============================================================

; --- NEW INT 08H (Timer Tick) - Сбор сенсоров ---
new_int08:
    pusha
    push ds
    push es
    
    ; DS = наш сегмент (COM файл)
    push cs
    pop ds
    
    ; Увеличить счётчик тиков
    inc word ptr [tick_counter]
    
    ; === СБОР СЕНСОРОВ (каждый тик) ===
    ; Сенсор 0: Время работы (тики / 256)
    mov ax, [tick_counter]
    mov [sensors+0], ax
    
    ; Сенсор 1: Свободная память (упрощённо через INT 21h/48h)
    ; (пропущено для экономии времени - можно добавить)
    
    ; Сенсор 2: Активность клавиатуры (флаг из INT 09h)
    ; ...
    
    ; Сенсоры 3-7: Можно читать порты, время и т.д.
    
    pop es
    pop ds
    popa
    
    ; Передать управление старому обработчику
    jmp dword ptr cs:[old_int08]

; --- NEW INT 09H (Keyboard) - Горячие клавиши ---
new_int09:
    in al, 60h               ; Считать скан-код
    
    ; Проверка ALT+S (скан-код 1Fh с ALT)
    test al, 80h             ; Key release?
    jnz skip_key_check
    cmp al, 1Fh              ; S key?
    jne skip_key_check
    mov ah, 12h              ; Проверить SHIFT status
    int 16h
    test al, 08h             ; ALT pressed?
    jz skip_key_check
    
    ; === ALT+S НАЖАТ - ПОКАЗАТЬ СТАТУС ===
    call show_status
    
skip_key_check:
    ; Передать старому обработчику
    jmp dword ptr cs:[old_int09]

; --- NEW INT 28H (DOS Idle) - "МЫСЛИТЕЛЬНЫЙ" ПРОЦЕСС ---
new_int28:
    pusha
    push ds
    push es
    
    push cs
    pop ds
    
    ; Проверить интервал "мышления"
    mov ax, [tick_counter]
    sub ax, [think_last]      ; think_last: переменная (добавить!)
    cmp ax, [think_interval]
    jl no_think_now
    
    ; Сохранить текущий тик
    mov ax, [tick_counter]
    mov [think_last], ax
    
    ; === ЗАПУСК НЕЙРОСЕТИ ===
    call run_inference
    
    ; === ПРИНЯТИЕ РЕШЕНИЯ ===
    call take_action
    
no_think_now:
    pop es
    pop ds
    popa
    
    ; Передать старому обработчику
    jmp dword ptr cs:[old_int28]

; Добавить переменную (если забыли выше):
think_last: dw 0

; ============================================================
;  RUN_INFERENCE - Прямой проход нейросети
;  Использует: sensors[], w_ih[], w_ho[], b_h[], b_o[]
;  Результат: output_act[]
; ============================================================
run_inference proc near
    push bp
    mov bp, sp
    push si
    push di
    push bx
    push cx
    push dx
    
    ; === СКРЫТЫЙ СЛОЙ ===
    xor di, di                ; di = индекс нейрона hidden
    
hidden_loop:
    xor si, si                ; si = индекс входа
    xor bx, bx                ; bx = аккумулятор (32-bit: DX:BX)
    xor dx, dx
    
input_loop:
    ; AX = sensors[si] * w_ih[si][di]
    mov ax, [sensors+si]      ; 16-bit input
    imul word ptr [w_ih+si*2+di*INPUT_SIZE*2]  ; DX:AX = product
    ; Сдвиг на FP_SHIFT (8) для fixed-point
    shrd bx, ax, 8
    shr dx, 8
    add bx, dx                ; BX += старшая часть
    
    inc si
    cmp si, INPUT_SIZE
    jl input_loop
    
    ; Добавить bias
    add bx, [b_h+di]
    
    ; Сигмоид через LUT
    ; Индекс = старший байт результата + 128 (смещение)
    mov ax, bx
    sar ax, 8                 ; Знаковый сдвиг
    and ax, 00FFh
    add al, 128
    xor bh, bh
    mov bl, al
    mov al, [sigmoid_lut+bx]  ; Значение из таблицы 0-255
    xor ah, ah
    shl ax, 8                 ; Преобразовать в Q8.8
    mov [hidden_act+di], ax   ; Сохранить активацию
    
    inc di
    cmp di, HIDDEN_SIZE
    jl hidden_loop
    
    === ВЫХОДНОЙ СЛОЙ ===
    xor di, di                ; di = индекс выхода
    
output_loop:
    xor si, si                ; si = индекс hidden
    xor bx, bx
    xor dx, dx
    
hidden_input_loop:
    mov ax, [hidden_act+si]
    imul word ptr [w_ho+si*2+di*HIDDEN_SIZE*2]
    shrd bx, ax, 8
    shr dx, 8
    add bx, dx
    
    inc si
    cmp si, HIDDEN_SIZE
    jl hidden_input_loop
    
    ; Bias
    add bx, [b_o+di]
    
    ; Сигмоид LUT
    mov ax, bx
    sar ax, 8
    and ax, 00FFh
    add al, 128
    xor bh, bh
    mov bl, al
    mov al, [sigmoid_lut+bx]
    xor ah, ah
    shl ax, 8
    mov [output_act+di], ax
    
    inc di
    cmp di, OUTPUT_SIZE
    jl output_loop
    
    ; === ГОТОВО ===
    pop dx
    pop cx
    pop bx
    pop di
    pop si
    pop bp
    ret
run_inference endp

; ============================================================
;  TAKE_ACTION - Выполнить действие на основе output_act[]
;  Действия: 0=Forward, 1=Backward, 2=Left, 3=Right
; ============================================================
take_action proc near
    ; Найти максимальный выход (argmax)
    xor si, si                ; si = best index
    mov di, 1                 ; di = current index
    mov bx, [output_act+0]    ; bx = best value
    
check_next:
    cmp di, OUTPUT_SIZE
    jge action_decided
    cmp bx, [output_act+di]
    jge not_better
    mov bx, [output_act+di]
    mov si, di
not_better:
    inc di
    jmp check_next
    
action_decided:
    ; SI теперь содержит выбранное действие
    ; Проверить cooldown
    cmp word ptr [action_cooldown], 0
    jg action_skip
    
    ; Установить cooldown (не действовать слишком часто)
    mov word ptr [action_cooldown], 36  ; ~2 секунды
    
    ; === ВЫПОЛНИТЬ ДЕЙСТВИЕ ===
    cmp si, 0
    je do_forward
    cmp si, 1
    je do_backward
    cmp si, 2
    je do_left
    cmp si, 3
    je do_right
    jmp action_done
    
do_forward:
    ; Действие "ВПЕРЁД" - пример: звуковой сигнал
    mov ax, 0E07h             ; Beep
    int 10h
    jmp action_done
    
do_backward:
    mov ax, 0E08h             ; Другой beep
    int 10h
    jmp action_done
    
do_left:
    mov ax, 0E09h
    int 10h
    jmp action_done
    
do_right:
    mov ax, 0E0Ah
    int 10h
    jmp action_done
    
action_skip:
    dec word ptr [action_cooldown]
    
action_done:
    ret
take_action endp

; ============================================================
;  SHOW_STATUS - Показать статус агента (ALT+S)
; ============================================================
show_status proc near
    pusha
    push ds
    push es
    
    ; Переключиться на видеопамять (или использовать INT 10h)
    ; Здесь упрощённо через DOS
    
    push cs
    pop ds
    
    ; Новая строка
    mov dl, 13
    mov ah, 02h
    int 21h
    mov dl, 10
    int 21h
    
    ; Вывод строки статуса
    mov dx, offset status_msg
    mov ah, 09h
    int 21h
    
    ; Вывод значений output_act (решения нейросети)
    mov cx, OUTPUT_SIZE
    xor si, si
    
out_loop:
    mov ax, [output_act+si]
    call print_fp_number
    
    ; Разделитель
    cmp si, OUTPUT_SIZE-1
    jge no_sep
    mov dl, '|'
    mov ah, 02h
    int 21h
no_sep:
    inc si
    loop out_loop
    
    ; Вывод выбранного действия
    mov dx, offset action_msg
    mov ah, 09h
    int 21h
    
    ; Найти argmax снова (можно кэшировать)
    call get_action_index
    add ax, ax              ; Word offset
    mov bx, ax
    mov dx, [action_names+bx]
    mov ah, 09h
    int 21h
    
    pop es
    pop ds
    popa
    ret
show_status endp

status_msg:  db 13,10,'=== SKYNET STATUS ===',13,10,'Outputs: $'
action_msg:  db 13,10,'Action: $'
action_names:
    dw offset act_fwd
    dw offset act_back
    dw offset act_left
    dw offset act_right
act_fwd:     db 'FORWARD$'
act_back:    db 'BACKWARD$'
act_left:    db 'LEFT$'
act_right:   db 'RIGHT$'

; ============================================================
;  HELPER: PRINT_FP_NUMBER - Вывод Q8.8 как decimal
;  Вход: AX = число в формате Q8.8
; ============================================================
print_fp_number proc near
    push ax
    push bx
    push cx
    push dx
    
    ; Целая часть
    sar ax, 8
    call print_word_dec
    
    ; Десятичная точка
    mov dl, '.'
    mov ah, 02h
    int 21h
    
    ; Дробная часть (2 знака после точки)
    pop ax                    ; Восстановить оригинальное значение
    push ax
    and ax, 00FFh             ; Маскировать дробную часть
    mov bl, 100
    mul bl                    ; AX = дробь * 100 / 256
    mov bl, 256
    div bl
    call print_byte_dec       ; Вывести 2 цифры
    
    pop dx
    pop cx
    pop bx
    pop ax
    ret
print_fp_number endp

; ============================================================
;  HELPERS: Числовой вывод (упрощённо)
; ============================================================
print_word_dec proc near
    ; AX = число для вывода (0-65535)
    push ax
    push bx
    push cx
    push dx
    
    mov bx, 10
    xor cx, cx                ; Счётчик цифр
    
div_loop:
    xor dx, dx
    div bx                    ; AX = quotient, DX = remainder
    push dx                   ; Сохранить цифру
    inc cx
    test ax, ax
    jnz div_loop
    
print_digits:
    pop dx
    add dl, '0'
    mov ah, 02h
    int 21h
    loop print_digits
    
    pop dx
    pop cx
    pop bx
    pop ax
    ret
print_word_dec endp

print_byte_dec proc near
    ; AL = 0-99
    push ax
    xor ah, ah
    mov bl, 10
    div bl                    ; AL = десятки, AH = единицы
    push ax
    add al, '0'
    mov dl, al
    mov ah, 02h
    int 21h
    pop ax
    add ah, '0'
    mov dl, ah
    mov ah, 02h
    int 21h
    pop ax
    ret
print_byte_dec endp

get_action_index proc near
    ; Возвращает индекс максимального output в AX
    xor si, si
    mov di, 1
    mov bx, [output_act+0]
    
gai_loop:
    cmp di, OUTPUT_SIZE
    jge gai_done
    cmp bx, [output_act+di]
    jge gai_next
    mov bx, [output_act+di]
    mov si, di
gai_next:
    inc di
    jmp gai_loop
gai_done:
    mov ax, si
    ret
get_action_index endp

; ============================================================
;  END OF PROGRAM MARKER (для вычисления размера TSR)
; ============================================================
end_of_program:

END START