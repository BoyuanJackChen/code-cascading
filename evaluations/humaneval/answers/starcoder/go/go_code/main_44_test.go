package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
         "strconv" 
 )

func TestChangeBase(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("22", ChangeBase(8, 3))
    assert.Equal("100", ChangeBase(9, 3))
    assert.Equal("11101010", ChangeBase(234, 2))
    assert.Equal("10000", ChangeBase(16, 2))
    assert.Equal("1000", ChangeBase(8, 2))
    assert.Equal("111", ChangeBase(7, 2))
    for i := 2; i < 8; i++ {
        assert.Equal(strconv.Itoa(i), ChangeBase(i, i+1))
    }
}
